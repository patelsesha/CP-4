import pandas as pd

df = pd.read_csv("malicious_phish_clean.csv")

print("=== DEFACEMENT SAMPLES ===")
print(df[df['type']=='defacement']['url'].sample(20, random_state=42).tolist())

print("\n=== DEFACEMENT PATTERNS ===")
def_urls = df[df['type']=='defacement']['url']
print("Has 'option=com':", round(def_urls.str.contains('option=com').mean()*100, 2), "%")
print("Has 'index.php' :", round(def_urls.str.contains('index.php').mean()*100, 2), "%")
print("Has 'view='     :", round(def_urls.str.contains('view=').mean()*100, 2), "%")
print("Has 'article'   :", round(def_urls.str.contains('article').mean()*100, 2), "%")


