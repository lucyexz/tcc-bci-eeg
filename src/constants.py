LABEL_COLUMN = "Values"
USER_COLUMN = "user"

# Colunas auxiliares criadas no pré-processamento. Não são features: guardam a
# que bloco de trial cada amostra pertence e sua posição temporal no sujeito.
BLOCK_COLUMN = "block"
POSITION_COLUMN = "position"

EEG_CHANNELS = ["AF3", "T7", "Pz", "T8", "AF4"]
EEG_BANDS = ["Theta", "Alpha", "BetaL", "BetaH", "Gamma"]

CLASS_NAMES = {0: "Esquerda", 1: "Direita", 2: "Neutro"}
