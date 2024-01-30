from .set_encoder_bert import BertSetEncoderMixin


class ElectraSetEncoderMixin(BertSetEncoderMixin):
    encoder_name = "electra"
