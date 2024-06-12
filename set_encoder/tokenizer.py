from lightning_ir.tokenizer.tokenizer import CrossEncoderTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerBase


class SetEncoderTokenizer(CrossEncoderTokenizer):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        query_length: int = 32,
        doc_length: int = 512,
        add_extra_token: bool = False,
        **kwargs,
    ):
        super().__init__(tokenizer, query_length, doc_length, **kwargs)
        self.interaction_token = "[INT]"
        if add_extra_token:
            self.add_tokens([self.interaction_token], special_tokens=True)
            self._tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $0 [SEP]",
                pair="[CLS] [INT] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.cls_token_id),
                    ("[SEP]", self.sep_token_id),
                    ("[INT]", self.interaction_token_id),
                ],
            )

    @property
    def interaction_token_id(self) -> int:
        if self.interaction_token in self.added_tokens_encoder:
            return self.added_tokens_encoder[self.interaction_token]
        raise ValueError(f"Token {self.interaction_token} not found in tokenizer")
