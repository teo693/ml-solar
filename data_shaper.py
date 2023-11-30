import numpy as np
import pandas as pd

# Załóżmy, że df to twój DataFrame
df = pd.read_csv('se2.csv')

# Wybierz kolumny, które chcesz uwzględnić w sekwencji
sequence_cols = ['state_id', 'old_state_id', 'attributes_id', 'last_updated_ts', 'last_changed_ts', 'metadata_id', 'attributes_id']

# Przekształć wybrane kolumny w sekwencje
sequences = [df[col].values.reshape(-1, 1) for col in sequence_cols]

# Sklej sekwencje w jeden trzywymiarowy tensor
data = np.concatenate(sequences, axis=1)

# Teraz data ma kształt (liczba próbek, liczba kroków czasowych, liczba cech)
print(data.shape)