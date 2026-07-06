import pandas as pd
from sklearn.model_selection import train_test_split

# Load your full ground truth
df = pd.read_csv("data/GroundTruth.csv")
df.columns = [c.strip().lower() for c in df.columns]

print(f"Total queries: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Check persona distribution
if 'persona' in df.columns and 'personas' not in df.columns:
    df = df.rename(columns={'persona': 'personas'})

print("\nPersona distribution:")
print(df['personas'].value_counts())

# Split: 80% train (for semantic router), 20% test (for evaluation)
# stratify ensures both splits have similar persona distribution
train_df, test_df = train_test_split(
    df, 
    test_size=0.3,      # 30% for testing
    random_state=42,    # reproducible split
    shuffle=True
)

# Save the splits
train_df.to_excel("GroundTruth_train.xlsx", index=False, engine="openpyxl")
test_df.to_excel("GroundTruth_test.xlsx", index=False, engine="openpyxl")

print(f"\n[DONE] Created GroundTruth_train.xlsx: {len(train_df)} queries")
print(f"[DONE] Created GroundTruth_test.xlsx: {len(test_df)} queries")
print(f"\nTrain/test ratio: {len(train_df)}/{len(test_df)}")