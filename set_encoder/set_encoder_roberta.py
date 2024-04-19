from .set_encoder_bert import BertSetEncoderMixin


class RoBERTaSetEncoderMixin(BertSetEncoderMixin):
    encoder_name = "roberta"
