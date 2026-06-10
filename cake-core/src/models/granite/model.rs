use anyhow::Result;
use async_trait::async_trait;

use super::block::GraniteBlock;
use super::history::GraniteHistory;
use crate::models::common::text_model::TextModelBase;
use crate::models::TextGenerator;
use crate::{
    cake::Context,
    models::{chat::Message, Generator, Token},
};

/// Default end of stream token if not found in configuration.
const DEFAULT_EOS_TOKEN: &str = "<|end_of_text|>";

/// IBM Granite main class (dense Granite 3.x and granite-4.0 non-hybrid).
pub struct Granite {
    base: TextModelBase,
    history: GraniteHistory,
}

#[async_trait]
impl Generator for Granite {
    type Shardable = GraniteBlock;
    const MODEL_NAME: &'static str = "granite";

    /// Load this model from the context.
    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let base = TextModelBase::load::<GraniteBlock>(ctx, DEFAULT_EOS_TOKEN).await?;
        Ok(Some(Box::new(Self {
            base,
            history: GraniteHistory::new(),
        })))
    }
}

#[async_trait]
impl TextGenerator for Granite {
    /// Add a message to the chat history.
    fn add_message(&mut self, message: Message) -> Result<()> {
        self.history.push(message);
        Ok(())
    }

    /// Reset the chat pipeline state.
    fn reset(&mut self) -> Result<()> {
        self.history.clear();
        self.base.reset();
        Ok(())
    }

    async fn goodbye(&mut self) -> Result<()> {
        self.base.goodbye().await
    }

    /// Return the next token.
    async fn next_token(&mut self, index: usize) -> Result<Token> {
        // Prefill tokens with chat history the first time.
        if self.base.generated == 0 {
            let dialog = self.history.encode_dialog_to_prompt();
            self.base.prepare_prompt(&dialog)?;
        }
        self.base.next_token(index).await
    }

    /// Return the number of generated tokens so far.
    fn generated_tokens(&self) -> usize {
        self.base.generated
    }
}
