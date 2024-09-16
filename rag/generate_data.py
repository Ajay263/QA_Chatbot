import random
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import uuid
from db import save_conversation, save_feedback, init_db
from assistant import GROQ_MODELS

# Initialize the database
init_db()

# Set the timezone
tz = ZoneInfo("Europe/Berlin")

# Define some sample data
courses = ["machine-learning-zoomcamp", "data-engineering-zoomcamp", "mlops-zoomcamp"]
questions = [
    "What is machine learning?",
    "How do I set up a data pipeline?",
    "What are the best practices for MLOps?",
    "Can you explain linear regression?",
    "What is the difference between SQL and NoSQL databases?",
]
relevance_options = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]

def generate_answer_data(question, course, model):
    return {
        "answer": f"Here's a synthetic answer to '{question}' for the {course} course.",
        "model_used": model,
        "response_time": random.uniform(0.5, 5.0),
        "relevance": random.choice(relevance_options),
        "relevance_explanation": "This is a synthetic relevance explanation.",
        "prompt_tokens": random.randint(50, 200),
        "completion_tokens": random.randint(100, 500),
        "total_tokens": random.randint(150, 700),
        "eval_prompt_tokens": random.randint(20, 100),
        "eval_completion_tokens": random.randint(50, 200),
        "eval_total_tokens": random.randint(70, 300),
        "openai_cost": random.uniform(0.0001, 0.01),
    }

def generate_historical_data():
    end_time = datetime.now(tz)
    start_time = end_time - timedelta(hours=6)
    current_time = start_time

    while current_time < end_time:
        conversation_id = str(uuid.uuid4())
        question = random.choice(questions)
        course = random.choice(courses)
        model = random.choice(GROQ_MODELS)
        answer_data = generate_answer_data(question, course, model)
        
        save_conversation(conversation_id, question, answer_data, course, current_time)
        
        # Add feedback for some conversations
        if random.random() < 0.7:  # 70% chance of feedback
            feedback = random.choice([-1, 1])
            feedback_time = current_time + timedelta(minutes=random.randint(1, 30))
            if feedback_time < end_time:
                save_feedback(conversation_id, feedback, feedback_time)
        
        current_time += timedelta(minutes=random.randint(5, 30))

def generate_live_data():
    while True:
        conversation_id = str(uuid.uuid4())
        question = random.choice(questions)
        course = random.choice(courses)
        model = random.choice(GROQ_MODELS)
        answer_data = generate_answer_data(question, course, model)
        
        current_time = datetime.now(tz)
        save_conversation(conversation_id, question, answer_data, course, current_time)
        
        # Add feedback for some conversations
        if random.random() < 0.7:  # 70% chance of feedback
            feedback = random.choice([-1, 1])
            feedback_time = current_time + timedelta(seconds=random.randint(30, 300))
            save_feedback(conversation_id, feedback, feedback_time)
        
        print(f"Generated data at {current_time}")
        time.sleep(1)  # Wait for 1 second before generating the next data point

if __name__ == "__main__":
    print("Generating historical data...")
    generate_historical_data()
    print("Historical data generation complete. Starting live data generation...")
    generate_live_data()