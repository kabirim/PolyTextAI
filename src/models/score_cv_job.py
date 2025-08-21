from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")  # Léger, CPU-friendly

def compute_similarity(list1, list2, threshold=0.4):
    if not list1 or not list2:
        return 0.0
    
    #Encode deux listes de textes en vecteurs
    embeddings1 = model.encode(list1, convert_to_tensor=True)
    embeddings2 = model.encode(list2, convert_to_tensor=True)
    
    #Calcule toutes les similarités cosinus
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    #Pour chaque élément de list1, garde la meilleure correspondance dans list2
    max_similarities = cosine_scores.max(dim=1).values
    
    # Applique un seuil : on ignore les similarités trop faibles
    filtered = [score.item() for score in max_similarities if score.item() >= threshold]
    
    # Si aucune similarité dépasse le seuil, on retourne 0
    if not filtered:
        return 0.0

    # Moyenne des similarités conservées
    avg_score = sum(filtered) / len(filtered)
    
    # Retourne le résultat en pourcentage
    return round(avg_score * 100, 2)

def match_years_of_experience(cv_years, jd_years):
    if jd_years <= 0:
        return 100.0  # aucune exigence = compatibilité max
    
    if cv_years <=0:
        return 0.0
    
    if cv_years >= jd_years:
        return 100.0   
   
    return round((cv_years / jd_years) * 100, 2)

def match_exact(value1, value2):
    
    # Support chaînes ou listes
    def extract(val):
        if isinstance(val, list):
            return val[0].strip().lower() if val else ""
        return val.strip().lower()

    v1 = extract(value1)
    v2 = extract(value2)
    
    if not v1 or not v2:
        return 0.0
    
    return 100.0 if v1 == v2 else 0.0

def match_any_overlap(list1, list2):
    if not list1 or not list2:
        return 0.0
    
    # Comparaison insensible à la casse
    set1 = set([item.strip().lower() for item in list1])
    set2 = set([item.strip().lower() for item in list2])
    
    # set1 & set2 retourne les éléments en commun
    # s’il y a au moins un match (par exemple "paris" dans les deux), on retourne 100.0
    if set1 & set2:
        return 100.0
    
    return 0.0