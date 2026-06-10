//! Granite chat template:
//!
//! ```text
//! <|start_of_role|>system<|end_of_role|>...<|end_of_text|>
//! <|start_of_role|>user<|end_of_role|>...<|end_of_text|>
//! <|start_of_role|>assistant<|end_of_role|>
//! ```

use crate::models::chat::Message;

const START_OF_ROLE: &str = "<|start_of_role|>";
const END_OF_ROLE: &str = "<|end_of_role|>";
const END_OF_TEXT: &str = "<|end_of_text|>";

/// Granite chat history encoder.
#[derive(Default)]
pub struct GraniteHistory {
    messages: Vec<Message>,
}

impl GraniteHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Encode the dialog into a prompt, ending with an open assistant turn.
    pub fn encode_dialog_to_prompt(&self) -> String {
        let mut prompt = String::new();
        for message in &self.messages {
            prompt.push_str(START_OF_ROLE);
            prompt.push_str(&message.role.to_string());
            prompt.push_str(END_OF_ROLE);
            prompt.push_str(&message.content);
            prompt.push_str(END_OF_TEXT);
            prompt.push('\n');
        }
        prompt.push_str(START_OF_ROLE);
        prompt.push_str("assistant");
        prompt.push_str(END_OF_ROLE);
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_history_opens_assistant_turn() {
        let history = GraniteHistory::new();
        assert_eq!(
            history.encode_dialog_to_prompt(),
            "<|start_of_role|>assistant<|end_of_role|>"
        );
    }

    #[test]
    fn test_single_user_message() {
        let mut history = GraniteHistory::new();
        history.push(Message::user("Hello".to_string()));
        assert_eq!(
            history.encode_dialog_to_prompt(),
            "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n\
             <|start_of_role|>assistant<|end_of_role|>"
        );
    }

    #[test]
    fn test_system_user_assistant_dialog() {
        let mut history = GraniteHistory::new();
        history.push(Message::system("Be brief.".to_string()));
        history.push(Message::user("Hi".to_string()));
        history.push(Message::assistant("Hello!".to_string()));
        history.push(Message::user("Bye".to_string()));
        assert_eq!(
            history.encode_dialog_to_prompt(),
            "<|start_of_role|>system<|end_of_role|>Be brief.<|end_of_text|>\n\
             <|start_of_role|>user<|end_of_role|>Hi<|end_of_text|>\n\
             <|start_of_role|>assistant<|end_of_role|>Hello!<|end_of_text|>\n\
             <|start_of_role|>user<|end_of_role|>Bye<|end_of_text|>\n\
             <|start_of_role|>assistant<|end_of_role|>"
        );
    }

    #[test]
    fn test_clear() {
        let mut history = GraniteHistory::new();
        history.push(Message::user("Hello".to_string()));
        history.clear();
        assert_eq!(
            history.encode_dialog_to_prompt(),
            "<|start_of_role|>assistant<|end_of_role|>"
        );
    }
}
