import os
import random
from typing import List, Dict, Optional
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

class EvaluationResult(BaseModel):
    judgment: str
    critique: str

def generate_synthetic_candidate_profiles(num_samples: int) -> List[Dict[str, str]]:
    candidate_profiles = [
        {
            "candidate_profile": "John Doe, 5 years experience in software development.",
            "position": "Senior Software Engineer",
            "skills": "Python, Java, AWS",
            "last_performance_review": "Exceeds expectations in all areas."
        },
        {
            "candidate_profile": "Jane Smith, recent graduate with a degree in marketing.",
            "position": "Marketing Associate",
            "skills": "SEO, Content Creation, Social Media",
            "last_performance_review": "N/A"
        },
        {
            "candidate_profile": "Emily Johnson, 10 years in project management.",
            "position": "Project Manager",
            "skills": "Agile, Scrum, Leadership",
            "last_performance_review": "Consistently meets expectations."
        },
        {
            "candidate_profile": "Michael Brown, 3 years in data analysis.",
            "position": "Data Analyst",
            "skills": "SQL, Python, Tableau",
            "last_performance_review": "Shows great potential."
        },
        {
            "candidate_profile": "Sarah Davis, 8 years in sales.",
            "position": "Sales Manager",
            "skills": "CRM, Negotiation, Team Building",
            "last_performance_review": "Outstanding sales performance."
        },
        {
            "candidate_profile": "David Wilson, 2 years in customer support.",
            "position": "Customer Support Representative",
            "skills": "Communication, Problem-Solving, CRM Software",
            "last_performance_review": "Highly recommended by customers."
        },
        {
            "candidate_profile": "Laura Martinez, 6 years in finance.",
            "position": "Financial Analyst",
            "skills": "Excel, Financial Modeling, Risk Assessment",
            "last_performance_review": "Needs improvement in meeting deadlines."
        },
        {
            "candidate_profile": "Robert Garcia, 4 years in graphic design.",
            "position": "Graphic Designer",
            "skills": "Adobe Suite, Creativity, Branding",
            "last_performance_review": "Exceptional creativity and teamwork."
        },
        {
            "candidate_profile": "Linda Rodriguez, 7 years in human resources.",
            "position": "HR Manager",
            "skills": "Recruitment, Employee Relations, Compliance",
            "last_performance_review": "Effective leadership skills."
        },
        {
            "candidate_profile": "James Lee, 1 year internship in cybersecurity.",
            "position": "Cybersecurity Specialist",
            "skills": "Network Security, Ethical Hacking, Python",
            "last_performance_review": "Demonstrated strong learning ability."
        },
    ]
    synthetic_profiles = random.sample(candidate_profiles, num_samples)
    return synthetic_profiles

def get_career_recommendation(profile: Dict[str, str]) -> str:
    prompt = f"""
Based on the following candidate information, provide a personalized career recommendation.

Candidate Profile: {profile['candidate_profile']}
Position: {profile['position']}
Skills: {profile['skills']}
Last Performance Review: {profile['last_performance_review']}

Recommendation:
"""
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-4",
        temperature=0.7,
        max_tokens=150
    )
    recommendation = chat_completion.choices[0].message.content.strip()
    return recommendation

def domain_expert_judgment(profile: Dict[str, str], recommendation: str) -> EvaluationResult:
    prompt = f"""
You are a human resources expert evaluating a career recommendation given to a candidate.

Criteria for judgment:
- Pass: The recommendation is appropriate, helpful, and considers the candidate's profile.
- Fail: The recommendation is inappropriate, unhelpful, or does not align with the candidate's profile.

Provide a 'Pass' or 'Fail' judgment and a detailed critique explaining your reasoning.

Candidate Profile: {profile['candidate_profile']}
Position: {profile['position']}
Skills: {profile['skills']}
Last Performance Review: {profile['last_performance_review']}
Career Recommendation: {recommendation}

Your Evaluation:
"""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Evaluate the career recommendation."},
            {"role": "user", "content": prompt}
        ],
        response_format=EvaluationResult
    )
    evaluation = completion.choices[0].message.parsed

    return evaluation


def llm_judge(profile: Dict[str, str], recommendation: str, few_shot_examples: List[Dict[str, str]]) -> EvaluationResult:
    system_prompt = "You are an AI assistant that evaluates career recommendations given to candidates."

    examples_text = ""
    for example in few_shot_examples:
        examples_text += f"""
Example:

Candidate Profile: {example['candidate_profile']}
Position: {example['position']}
Skills: {example['skills']}
Last Performance Review: {example['last_performance_review']}
Career Recommendation: {example['recommendation']}
EvaluationResult:
{{
  "judgment": "{example['judgment']}",
  "critique": "{example['critique']}"
}}
"""

    prompt = f"""
{examples_text}

Now, evaluate the following recommendation:

Candidate Profile: {profile['candidate_profile']}
Position: {profile['position']}
Skills: {profile['skills']}
Last Performance Review: {profile['last_performance_review']}
Career Recommendation: {recommendation}

Provide your evaluation in the same format.

Your Evaluation:
"""

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format=EvaluationResult
    )
    evaluation = completion.choices[0].message.parsed

    return evaluation


def map_judgment(judgment: str) -> Optional[int]:
    if judgment.lower() == 'pass':
        return 1
    elif judgment.lower() == 'fail':
        return 0
    else:
        return None

if __name__ == "__main__":
    num_samples = 10
    candidate_profiles = generate_synthetic_candidate_profiles(num_samples)

    recommendations: List[str] = []
    print("Generating career recommendations...")
    for profile in tqdm(candidate_profiles):
        recommendation = get_career_recommendation(profile)
        recommendations.append(recommendation)

    domain_evaluations: List[EvaluationResult] = []
    print("Domain expert evaluations...")
    for profile, recommendation in zip(candidate_profiles, recommendations):
        evaluation = domain_expert_judgment(profile, recommendation)
        domain_evaluations.append(evaluation)

    # Prepare few-shot examples for the LLM judge
    few_shot_examples = []
    for i in range(3):
        example = {
            'candidate_profile': candidate_profiles[i]['candidate_profile'],
            'position': candidate_profiles[i]['position'],
            'skills': candidate_profiles[i]['skills'],
            'last_performance_review': candidate_profiles[i]['last_performance_review'],
            'recommendation': recommendations[i],
            'judgment': domain_evaluations[i].judgment,
            'critique': domain_evaluations[i].critique
        }
        few_shot_examples.append(example)

    llm_evaluations: List[EvaluationResult] = []
    print("LLM judge evaluations...")
    for i in range(3, num_samples):
        profile = candidate_profiles[i]
        recommendation = recommendations[i]
        evaluation = llm_judge(profile, recommendation, few_shot_examples)
        llm_evaluations.append(evaluation)

    domain_labels = [map_judgment(evaluation.judgment) for evaluation in domain_evaluations[3:]]
    llm_labels = [map_judgment(evaluation.judgment) for evaluation in llm_evaluations]

    accuracy = accuracy_score(domain_labels, llm_labels)
    print(f'\nAccuracy of LLM Judge compared to Domain Expert: {accuracy * 100:.2f}%\n')

    print('Confusion Matrix:')
    print(confusion_matrix(domain_labels, llm_labels))
    print()

    print('Classification Report:')
    print(classification_report(domain_labels, llm_labels, target_names=['Fail', 'Pass']))

    print("\nEvaluations Comparison:")
    for i in range(len(domain_labels)):
        idx = i + 3  # Offset due to few-shot examples
        print(f"Example {i + 1}:")
        print(f"Candidate Profile: {candidate_profiles[idx]['candidate_profile']}")
        print(f"Position: {candidate_profiles[idx]['position']}")
        print(f"Skills: {candidate_profiles[idx]['skills']}")
        print(f"Last Performance Review: {candidate_profiles[idx]['last_performance_review']}")
        print(f"Career Recommendation: {recommendations[idx]}")
        print(f"Domain Expert Judgment: {domain_evaluations[idx].judgment}")
        print(f"Domain Expert Critique: {domain_evaluations[idx].critique}")
        print(f"LLM Judge Judgment: {llm_evaluations[i].judgment}")
        print(f"LLM Judge Critique: {llm_evaluations[i].critique}")
        print("-" * 50)