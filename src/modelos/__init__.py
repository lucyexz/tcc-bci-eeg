"""Os cinco modelos do trabalho.

`MODELOS[nome]` = (classe de configuração, função que monta o modelo, reshape da janela).
"""

from src.modelos import cnn, cnn_transformer, recorrentes, transformer

MODELOS = {
    "cnn": (cnn.CNNConfig, cnn.build, cnn.reshape),
    "bilstm": (recorrentes.BiLSTMConfig, recorrentes.build_bilstm, recorrentes.reshape),
    "bigru": (recorrentes.BiGRUConfig, recorrentes.build_bigru, recorrentes.reshape),
    "transformer": (transformer.TransformerConfig, transformer.build, transformer.reshape),
    "cnn_transformer": (cnn_transformer.CNNTransformerConfig, cnn_transformer.build,
                        cnn_transformer.reshape),
}

NOMES = {
    "cnn": "CNN",
    "bilstm": "Bi-LSTM",
    "bigru": "Bi-GRU",
    "transformer": "Transformer",
    "cnn_transformer": "CNN + Transformer",
}
