from collections import Counter



def count_skills(path, skills): 
    try:
        with open(path,encoding='utf-8') as f:
            text = f.read().lower()
            text = text.lower()
    except FileNotFoundError:
        print(f"[Warming] {path} not found,skipping")
        return Counter()

    return Counter({skill: text.count(skill.lower()) for skill in skills})


my_skills = ["Python", "SQL","Power BI","Excel","Machine Learning","Statistics","Numpy","Deep Learning"]

#count the skiills in each JD file (a list comprehension buuld the 3 counter).
counts = [count_skills(f"data/jd{i}.txt", my_skills) for i in (1,2,3)]

for i,c in enumerate(counts,start =1):
    print(f"jd{i}.text -> {dict(c)}")

study_list = set().union(*[set(c.keys()) for c in counts])

print("\nStudy list (skills that are not in all JDs):")
print(sorted(study_list))
    