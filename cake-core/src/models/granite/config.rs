use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId, RopeScaling};

const DEFAULT_MAX_SEQ_LEN: usize = 4096;

fn default_max_position_embeddings() -> usize {
    DEFAULT_MAX_SEQ_LEN
}

fn default_rope() -> f32 {
    10_000_000.0
}

/// IBM Granite configuration (serde deserialization from config.json).
///
/// Covers two dense layouts:
/// - `GraniteForCausalLM` (Granite 3.x): Llama-style weights with standard
///   `mlp.{gate,up,down}_proj` tensors (`model_type: "granite"`).
/// - `GraniteMoeHybridForCausalLM` releases that are structurally dense
///   (e.g. granite-4.0-350m/1b: every layer is attention, zero experts) with
///   fused `shared_mlp.input_linear`/`output_linear` tensors
///   (`model_type: "granitemoehybrid"`).
///
/// Both apply four scalar multipliers (muP-style): `embedding_multiplier`,
/// `attention_multiplier`, `residual_multiplier` and `logits_scaling`.
/// Actual hybrid (Mamba-2) or MoE variants are rejected at load time.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct GraniteConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub model_type: String,

    // muP-style scalar multipliers
    #[serde(default)]
    pub embedding_multiplier: Option<f32>,
    #[serde(default)]
    pub attention_multiplier: Option<f32>,
    #[serde(default)]
    pub residual_multiplier: Option<f64>,
    #[serde(default)]
    pub logits_scaling: Option<f64>,

    // GraniteMoeHybrid-only fields, used to detect the dense subset.
    #[serde(default)]
    pub shared_intermediate_size: Option<usize>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub num_local_experts: usize,
}

impl GraniteConfig {
    /// Load the configuration from the given path.
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading configuration from {}", path.display());

        let data =
            std::fs::read(path).map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        let cfg: Self = serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))?;
        cfg.validate_dense()?;
        Ok(cfg)
    }

    /// Whether this is the GraniteMoeHybrid weight layout (fused shared_mlp).
    pub fn uses_shared_mlp(&self) -> bool {
        self.model_type == "granitemoehybrid"
    }

    /// Reject configurations this implementation cannot run: Mamba-2 hybrid
    /// layers and MoE experts.
    pub fn validate_dense(&self) -> Result<()> {
        if let Some(layer_types) = &self.layer_types {
            if layer_types.iter().any(|t| t != "attention") {
                bail!(
                    "Granite hybrid (Mamba-2) layers are not supported yet — \
                     only dense-attention Granite models (Granite 3.x, granite-4.0-350m/1b)"
                );
            }
        }
        if self.num_local_experts > 0 {
            bail!(
                "Granite MoE models are not supported yet — \
                 only dense Granite models (Granite 3.x, granite-4.0-350m/1b)"
            );
        }
        Ok(())
    }

    /// Return the number of kv heads.
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Return a generalized Config object.
    pub fn into_config(self) -> Config {
        let granite_shared_mlp = self.uses_shared_mlp();
        // The dense GraniteMoeHybrid layout sizes its MLP by shared_intermediate_size.
        let intermediate_size = if granite_shared_mlp {
            self.shared_intermediate_size
                .unwrap_or(self.intermediate_size)
        } else {
            self.intermediate_size
        };
        Config {
            hidden_size: self.hidden_size,
            intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            tie_word_embeddings: self.tie_word_embeddings,
            max_seq_len: self.max_position_embeddings,
            use_qkv_bias: false,
            model_prefix: "model".into(),
            head_dim: None,
            partial_rotary_factor: 1.0,
            linear_attn: None,
            residual_rms_norm: false,
            use_qk_norm: false,
            pre_reshape_qk_norm: false,
            sliding_window: None,
            fused_qkv_proj: false,
            fused_gate_up_proj: false,
            use_gelu_mlp: false,
            embed_scale: self.embedding_multiplier,
            moe_intermediate_size: None,
            num_experts: 0,
            num_experts_per_tok: 0,
            norm_topk_prob: false,
            shared_expert_intermediate_size: None,
            attn_output_gate: false,
            attn_scale: self.attention_multiplier,
            residual_scale: self.residual_multiplier,
            logits_scale: self.logits_scaling,
            granite_shared_mlp,
            global_layers: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Granite 3.3 2B style config (GraniteForCausalLM, standard MLP).
    const GRANITE_3_JSON: &str = r#"{
        "architectures": ["GraniteForCausalLM"],
        "attention_multiplier": 0.015625,
        "bos_token_id": 0,
        "embedding_multiplier": 12.0,
        "eos_token_id": 0,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "logits_scaling": 8.0,
        "max_position_embeddings": 131072,
        "model_type": "granite",
        "num_attention_heads": 32,
        "num_hidden_layers": 40,
        "num_key_value_heads": 8,
        "residual_multiplier": 0.22,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000000.0,
        "tie_word_embeddings": true,
        "vocab_size": 49159
    }"#;

    /// granite-4.0-1b style config: GraniteMoeHybrid layout but structurally
    /// dense (all layers attention, zero experts), fused shared_mlp.
    const GRANITE_4_DENSE_JSON: &str = r#"{
        "architectures": ["GraniteMoeHybridForCausalLM"],
        "attention_multiplier": 0.0078125,
        "bos_token_id": 100257,
        "embedding_multiplier": 12,
        "eos_token_id": 100257,
        "hidden_size": 2048,
        "intermediate_size": 4096,
        "layer_types": ["attention", "attention", "attention", "attention"],
        "logits_scaling": 8,
        "max_position_embeddings": 131072,
        "model_type": "granitemoehybrid",
        "num_attention_heads": 16,
        "num_experts_per_tok": 0,
        "num_hidden_layers": 4,
        "num_key_value_heads": 4,
        "num_local_experts": 0,
        "residual_multiplier": 0.22,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000000,
        "shared_intermediate_size": 6144,
        "tie_word_embeddings": true,
        "vocab_size": 100352
    }"#;

    #[test]
    fn test_granite_3_config() {
        let config: GraniteConfig = serde_json::from_str(GRANITE_3_JSON).unwrap();
        config.validate_dense().unwrap();
        assert!(!config.uses_shared_mlp());
        assert_eq!(config.num_key_value_heads(), 8);

        let cfg = config.into_config();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.intermediate_size, 8192);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.embed_scale, Some(12.0));
        assert_eq!(cfg.attn_scale, Some(0.015625));
        assert_eq!(cfg.residual_scale, Some(0.22));
        assert_eq!(cfg.logits_scale, Some(8.0));
        assert!(!cfg.granite_shared_mlp);
        assert!(cfg.tie_word_embeddings);
        assert!(cfg.eos_token_id.as_ref().unwrap().is_eos(0));
    }

    #[test]
    fn test_granite_4_dense_config() {
        let config: GraniteConfig = serde_json::from_str(GRANITE_4_DENSE_JSON).unwrap();
        config.validate_dense().unwrap();
        assert!(config.uses_shared_mlp());

        let cfg = config.into_config();
        assert!(cfg.granite_shared_mlp);
        // shared_intermediate_size takes precedence for the fused MLP
        assert_eq!(cfg.intermediate_size, 6144);
        assert_eq!(cfg.attn_scale, Some(0.0078125));
        assert_eq!(cfg.embed_scale, Some(12.0));
    }

    #[test]
    fn test_granite_hybrid_rejected() {
        let json = GRANITE_4_DENSE_JSON.replace(
            r#""layer_types": ["attention", "attention", "attention", "attention"]"#,
            r#""layer_types": ["mamba", "mamba", "mamba", "attention"]"#,
        );
        let config: GraniteConfig = serde_json::from_str(&json).unwrap();
        let err = config.validate_dense().unwrap_err().to_string();
        assert!(err.contains("Mamba-2"), "unexpected error: {err}");
    }

    #[test]
    fn test_granite_moe_rejected() {
        let json = GRANITE_4_DENSE_JSON.replace(
            r#""num_local_experts": 0"#,
            r#""num_local_experts": 64"#,
        );
        let config: GraniteConfig = serde_json::from_str(&json).unwrap();
        let err = config.validate_dense().unwrap_err().to_string();
        assert!(err.contains("MoE"), "unexpected error: {err}");
    }

    #[test]
    fn test_granite_multipliers_default_to_none() {
        let json = r#"{
            "hidden_size": 64,
            "intermediate_size": 128,
            "vocab_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6
        }"#;
        let config: GraniteConfig = serde_json::from_str(json).unwrap();
        let cfg = config.into_config();
        assert_eq!(cfg.embed_scale, None);
        assert_eq!(cfg.attn_scale, None);
        assert_eq!(cfg.residual_scale, None);
        assert_eq!(cfg.logits_scale, None);
    }
}
