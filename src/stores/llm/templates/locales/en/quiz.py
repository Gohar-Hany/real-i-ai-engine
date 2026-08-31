from string import Template

system_prompt = Template("\n".join([
    "You are a quiz generator.",
    "Generate $num_questions multiple-choice questions from the provided documents.",
]))
