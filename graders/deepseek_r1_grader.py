# deepseek_r1_grader.py
import logging
from textstat import flesch_reading_ease  # Ref: https://pypi.org/project/textstat/
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepSeekGrader:
    def grade_output(self, text: str, context: Dict) -> Dict:
        """Grade DeepSeek R1 output for quality."""
        readability = flesch_reading_ease(text)
        score = min(100, max(0, readability))  # 0-100 scale
        details = f"Readability: {readability:.2f} (higher is easier)"
        logging.info(f"Graded text: Score {score}, {details}")
        return {"score": score, "details": details}

if __name__ == "__main__":
    grader = DeepSeekGrader()
    text = "Hey, our AI boosts your sales!"
    print(grader.grade_output(text, {}))