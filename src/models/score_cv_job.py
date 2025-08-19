from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")  # Léger, CPU-friendly

def compute_similarity(list1, list2):
    if not list1 or not list2:
        return 0.0
    #Encode deux listes de textes en vecteurs
    embeddings1 = model.encode(list1, convert_to_tensor=True)
    embeddings2 = model.encode(list2, convert_to_tensor=True)
    
    #Calcule toutes les similarités cosinus
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    #Pour chaque élément de list1, garde la meilleure correspondance dans list2
    max_similarities = cosine_scores.max(dim=1).values
    
    #Fait la moyenne de ces correspondances
    avg_score = max_similarities.mean().item()
    
    #Retourne le résultat en pourcentage
    return round(avg_score * 100, 2)

def match_years_of_experience(cv_years, jd_years):
    if cv_years >= jd_years:
        return 100.0
    return round((cv_years / jd_years) * 100, 2)

def match_exact(value1, value2):
    return 100.0 if value1.lower() == value2.lower() else 0.0