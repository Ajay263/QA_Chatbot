import streamlit as st
import time
import uuid
import os
from dotenv import load_dotenv
from assistant import get_answer
from db import (
    save_conversation,
    save_feedback,
    get_recent_conversations,
    get_feedback_stats,
)

# Load environment variables from .env file
load_dotenv()

GROQ_MODELS = ["mixtral-8x7b-32768", "llama-3.1-70b-versatile", "gemma2-9b-it", "llama3-70b-8192"]

def print_log(message):
    """Print log messages with a newline and flush."""
    print(message, flush=True)

def main():
    print_log("Starting the Course Assistant application")
    st.title("Course Assistant")

    # Session state initialization
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
        print_log(f"New conversation started with ID: {st.session_state.conversation_id}")
    if "count" not in st.session_state:
        st.session_state.count = 0
        print_log("Feedback count initialized to 0")
    if "answer_given" not in st.session_state:
        st.session_state.answer_given = False
    if "feedback_count" not in st.session_state:  # Initialize feedback count
        st.session_state.feedback_count = 0
    if "feedback_message" not in st.session_state:  # Initialize feedback message
        st.session_state.feedback_message = ""

    # Course selection
    course = st.selectbox(
        "Select a course:",
        ["machine-learning-zoomcamp", "data-engineering-zoomcamp", "mlops-zoomcamp"],
    )
    print_log(f"User selected course: {course}")

    # Search type selection
    search_type = st.radio("Select search type:", ["Text", "Vector"])
    print_log(f"User selected search type: {search_type}")

    # Model selection
    model = st.selectbox("Select model:", GROQ_MODELS)
    print_log(f"User selected model: {model}")

    # User input
    user_input = st.text_input("Enter your question:")

    if st.button("Ask"):
        if not user_input:
            st.warning("Please enter a question before clicking 'Ask'.")
        else:
            print_log(f"User asked: '{user_input}'")
            with st.spinner("Processing..."):
                try:
                    print_log(f"Getting answer from assistant using model {model} and {search_type} search")
                    start_time = time.time()
                    answer_data = get_answer(user_input, course, search_type, model)
                    end_time = time.time()
                    print_log(f"Answer received in {end_time - start_time:.2f} seconds")

                    if isinstance(answer_data, dict):
                        if 'error' in answer_data:
                            st.error(f"An error occurred: {answer_data['error']}")
                            print_log(f"Error occurred: {answer_data['error']}")
                        else:
                            st.success("Answer generated successfully!")
                            st.markdown(answer_data["answer"])

                            # Display monitoring information
                            st.markdown("**Answer Metadata:**")
                            st.markdown(f"- Response time: {answer_data['response_time']:.2f} seconds")
                            st.markdown(f"- Relevance: {answer_data['relevance']}")
                            st.markdown(f"- Model used: {answer_data['model_used']}")
                            st.markdown(f"- Prompt tokens: {answer_data['prompt_tokens']}")
                            st.markdown(f"- Completion tokens: {answer_data['completion_tokens']}")
                            st.markdown(f"- Total tokens: {answer_data['total_tokens']}")
                            st.markdown(f"- OpenAI equivalent cost: ${answer_data['openai_cost']:.5f}")

                            # Save conversation to database
                            print_log("Saving conversation to database")
                            try:
                                save_conversation(
                                    st.session_state.conversation_id, user_input, answer_data, course
                                )
                                print_log("Conversation saved successfully")
                            except Exception as e:
                                print_log(f"Error saving conversation: {str(e)}")

                            # Set answer_given to True to allow feedback
                            st.session_state.answer_given = True

                    else:
                        st.warning("Unexpected response format. Please try again.")
                        print_log("Error: answer_data is not a dictionary")
                except Exception as e:
                    st.error("An unexpected error occurred. Please try again later.")
                    print_log(f"Exception occurred: {str(e)}")

    if st.session_state.answer_given:
        st.write("Please provide feedback on the answer:")
        col1, col2 = st.columns(2)

        def on_feedback_click(value):
            st.session_state.feedback_count += value
            try:
                save_feedback(st.session_state.conversation_id, value)
                st.session_state.feedback_message = "Thank you for your feedback!"
            except Exception as e:
                print_log(f"Error saving feedback: {str(e)}")
                st.session_state.feedback_message = "Your feedback was recorded but couldn't be saved. Thank you!"
            st.session_state.answer_given = False  # Reset for next question

        with col1:
            if st.button("👍 Thumbs Up", on_click=on_feedback_click, args=(1,)):
                pass
        with col2:
            if st.button("👎 Thumbs Down", on_click=on_feedback_click, args=(-1,)):
                pass

        if st.session_state.feedback_message:
            st.success(st.session_state.feedback_message)
            st.session_state.feedback_message = ""  # Clear the message after displaying

    # Display recent conversations
    st.subheader("Recent Conversations")
    relevance_filter = st.selectbox("Filter by relevance:", ["All", "RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"])
    recent_conversations = get_recent_conversations(limit=5, relevance=relevance_filter if relevance_filter != "All" else None)
    for conv in recent_conversations:
        st.markdown(f"**Question:** {conv['question']}")
        st.markdown(f"**Answer:** {conv['answer']}")
        st.markdown(f"**Relevance:** {conv['relevance']}")
        st.markdown(f"**Model:** {conv['model_used']}")
        st.markdown(f"**Response Time:** {conv['response_time']:.2f} seconds")
        st.markdown(f"**Total Tokens:** {conv['total_tokens']}")
        st.markdown(f"**OpenAI Cost:** ${conv['openai_cost']:.5f}")
        st.markdown("---")

    # Display feedback stats
    feedback_stats = get_feedback_stats()
    st.subheader("Feedback Statistics")
    st.markdown(f"👍 Thumbs Up: {feedback_stats['thumbs_up']}")
    st.markdown(f"👎 Thumbs Down: {feedback_stats['thumbs_down']}")

if __name__ == "__main__":
    print_log("Course Assistant application started")
    main()