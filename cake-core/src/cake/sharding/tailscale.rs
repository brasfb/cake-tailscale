//! Tailscale-aware worker discovery.
//!
//! UDP broadcast does not cross a tailnet, but unicast UDP does — and workers
//! already answer discovery queries unicast to the query's source address
//! (see [`super::discovery`]). So Tailscale discovery only needs the tailnet
//! peer list: the master sends the regular discovery query unicast to each
//! peer instead of relying on broadcast. The peer list comes from the local
//! Tailscale daemon via `tailscale status --json`.

use std::net::Ipv4Addr;

use anyhow::{Result, anyhow};
use serde::Deserialize;

/// Path of the CLI bundled inside the macOS GUI app, which is not on PATH
/// by default.
const MACOS_APP_CLI: &str = "/Applications/Tailscale.app/Contents/MacOS/Tailscale";

/// A peer node in the tailnet, as reported by `tailscale status --json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TailscalePeer {
    pub ip: Ipv4Addr,
    pub hostname: String,
    pub online: bool,
    pub os: String,
}

#[derive(Deserialize)]
struct StatusJson {
    #[serde(rename = "Peer", default)]
    peer: std::collections::HashMap<String, PeerJson>,
}

#[derive(Deserialize)]
struct PeerJson {
    #[serde(rename = "HostName", default)]
    host_name: String,
    #[serde(rename = "TailscaleIPs", default)]
    tailscale_ips: Vec<String>,
    #[serde(rename = "Online", default)]
    online: bool,
    #[serde(rename = "OS", default)]
    os: String,
}

/// Parse `tailscale status --json` output into a peer list.
///
/// Peers without an IPv4 tailnet address are skipped (cake's discovery
/// protocol is IPv4-only).
pub fn parse_status_json(json: &str) -> Result<Vec<TailscalePeer>> {
    let status: StatusJson = serde_json::from_str(json)
        .map_err(|e| anyhow!("failed to parse tailscale status JSON: {}", e))?;

    let mut peers: Vec<TailscalePeer> = status
        .peer
        .into_values()
        .filter_map(|p| {
            let ip = p
                .tailscale_ips
                .iter()
                .find_map(|s| s.parse::<Ipv4Addr>().ok())?;
            Some(TailscalePeer {
                ip,
                hostname: p.host_name,
                online: p.online,
                os: p.os,
            })
        })
        .collect();

    // HashMap iteration order is random — keep results deterministic.
    peers.sort_by(|a, b| a.ip.cmp(&b.ip));
    Ok(peers)
}

/// Run the Tailscale CLI and return the raw `status --json` output.
///
/// Tries `tailscale` from PATH first (Linux tailscaled, macOS standalone),
/// then the binary bundled in the macOS GUI app.
pub fn tailscale_status() -> Result<String> {
    for cmd in ["tailscale", MACOS_APP_CLI] {
        match std::process::Command::new(cmd)
            .args(["status", "--json"])
            .output()
        {
            Ok(out) if out.status.success() => {
                return Ok(String::from_utf8_lossy(&out.stdout).into_owned());
            }
            Ok(out) => {
                return Err(anyhow!(
                    "'{} status --json' failed: {}",
                    cmd,
                    String::from_utf8_lossy(&out.stderr).trim()
                ));
            }
            Err(_) => continue, // binary not found — try the next location
        }
    }
    Err(anyhow!(
        "tailscale CLI not found — install Tailscale, or symlink the macOS app CLI: \
         ln -s {} /usr/local/bin/tailscale",
        MACOS_APP_CLI
    ))
}

/// Return the IPv4 addresses of all online tailnet peers.
pub fn tailnet_peer_ips() -> Result<Vec<Ipv4Addr>> {
    let peers = parse_status_json(&tailscale_status()?)?;
    Ok(peers.into_iter().filter(|p| p.online).map(|p| p.ip).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer_json(hostname: &str, ips: &[&str], online: bool) -> String {
        format!(
            r#""nodekey:{}": {{"HostName": "{}", "TailscaleIPs": [{}], "Online": {}, "OS": "macOS"}}"#,
            hostname,
            hostname,
            ips.iter()
                .map(|ip| format!("\"{}\"", ip))
                .collect::<Vec<_>>()
                .join(", "),
            online
        )
    }

    fn status_json(peers: &[String]) -> String {
        format!(r#"{{"Version": "1.80.0", "Peer": {{{}}}}}"#, peers.join(", "))
    }

    #[test]
    fn test_parse_no_peers() {
        let peers = parse_status_json(r#"{"Version": "1.80.0"}"#).unwrap();
        assert!(peers.is_empty());
        let peers = parse_status_json(r#"{"Version": "1.80.0", "Peer": {}}"#).unwrap();
        assert!(peers.is_empty());
    }

    #[test]
    fn test_parse_single_peer() {
        let json = status_json(&[peer_json("mini2", &["100.101.102.103", "fd7a::1"], true)]);
        let peers = parse_status_json(&json).unwrap();
        assert_eq!(
            peers,
            vec![TailscalePeer {
                ip: "100.101.102.103".parse().unwrap(),
                hostname: "mini2".to_string(),
                online: true,
                os: "macOS".to_string(),
            }]
        );
    }

    #[test]
    fn test_parse_multiple_peers_sorted_by_ip() {
        let json = status_json(&[
            peer_json("mini4", &["100.64.0.4"], true),
            peer_json("mini2", &["100.64.0.2"], true),
            peer_json("mini3", &["100.64.0.3"], false),
        ]);
        let peers = parse_status_json(&json).unwrap();
        assert_eq!(peers.len(), 3);
        let hostnames: Vec<&str> = peers.iter().map(|p| p.hostname.as_str()).collect();
        assert_eq!(hostnames, vec!["mini2", "mini3", "mini4"]);
    }

    #[test]
    fn test_parse_skips_ipv6_only_peer() {
        let json = status_json(&[
            peer_json("v6only", &["fd7a:115c:a1e0::1"], true),
            peer_json("mini2", &["100.64.0.2"], true),
        ]);
        let peers = parse_status_json(&json).unwrap();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].hostname, "mini2");
    }

    #[test]
    fn test_parse_ipv6_listed_first() {
        // IPv4 should be found even when not the first address.
        let json = status_json(&[peer_json("mini2", &["fd7a::1", "100.64.0.2"], true)]);
        let peers = parse_status_json(&json).unwrap();
        assert_eq!(peers[0].ip, "100.64.0.2".parse::<Ipv4Addr>().unwrap());
    }

    #[test]
    fn test_parse_missing_fields_defaulted() {
        let json = r#"{"Peer": {"nodekey:x": {"TailscaleIPs": ["100.64.0.9"]}}}"#;
        let peers = parse_status_json(json).unwrap();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].hostname, "");
        assert!(!peers[0].online);
    }

    #[test]
    fn test_parse_malformed_json_errors() {
        assert!(parse_status_json("not json").is_err());
        assert!(parse_status_json(r#"{"Peer": []}"#).is_err());
    }

    #[test]
    fn test_offline_peers_filtered_by_caller() {
        let json = status_json(&[
            peer_json("up", &["100.64.0.1"], true),
            peer_json("down", &["100.64.0.5"], false),
        ]);
        let peers = parse_status_json(&json).unwrap();
        let online_ips: Vec<Ipv4Addr> =
            peers.into_iter().filter(|p| p.online).map(|p| p.ip).collect();
        assert_eq!(online_ips, vec!["100.64.0.1".parse::<Ipv4Addr>().unwrap()]);
    }
}
