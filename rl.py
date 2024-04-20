import torch
from trl import PPOTrainer, PPOConfig, create_reference_model, AutoModelForCausalLMWithValueHead
from qna import *
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
import streamlit as st 



def reinforcement(text, user_choice):
    ppo_config = PPOConfig(batch_size=1, mini_batch_size=1)
    model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
    model_ref = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  

    ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer=tokenizer)

    # Main loop of the application
    while True:
        input_query = '''Generate a JSON representation of all the available data.

{
    "data": [
        {
            "attribute": "value",
            "attribute": "value",
            ...
        },
        {
            "attribute": "value",
            "attribute": "value",
            ...
        },
        ...
    ]
}'''
        vectorstore, llm, embeddings_hf = processing_embedding(text)
        response_1, response_2= answer_question(vectorstore, llm, input_query)
        

        # col1, col2, col3 = st.columns(3)

        # with col1:
        #     st.write(response_1)

        # with col2:
        #     st.write(response_2)

        # with col3:
        #     genre = st.radio(
        #         "Which response is better (1/2)?",
        #         ["1", "2"])


        # user_choice = input("Which response is better? (1/2): ")
        input_query = tokenizer.encode(str(input_query), return_tensors="pt")
        response_1_new = tokenizer.encode(str(response_1), return_tensors="pt")
        response_2_new = tokenizer.encode(str(response_2), return_tensors="pt")

        if user_choice == "1":
            reward = [torch.tensor(1.0)]
            train_stats = ppo_trainer.step([input_query[0]], [response_1_new[0]], reward)
            
        elif user_choice == "2":
            reward = [torch.tensor(1.0)]
            train_stats = ppo_trainer.step([input_query[0]], [response_2_new[0]], reward)
        else:
            print("Invalid choice. Skipping model update.")

        # Repeat the loop for the next query





















































# import torch
# from your_llm_module import generate_response  # Replace with your LLM module or function
# from trl import PPOTrainer, PPOConfig, create_reference_model

# # Initialize PPO Trainer
# ppo_config = PPOConfig(batch_size=1, mini_batch_size=1)
# model = create_reference_model()  # Initialize your LLM model
# model_ref = create_reference_model(model)
# ppo_trainer = PPOTrainer(ppo_config, model, model_ref)

# # Main loop of the application
# while True:
#     # Step 1: Generate responses
#     input_query = input("Enter your query: ")
#     response_1 = generate_response(input_query)
#     response_2 = generate_response(input_query)

#     # Step 2: Display responses
#     print("Response 1:", response_1)
#     print("Response 2:", response_2)

#     # Step 3: Collect feedback
#     user_choice = input("Which response is better? (1/2): ")

#     # Step 4: Update model
#     if user_choice == "1":
#         reward = [torch.tensor(1.0)]
#         train_stats = ppo_trainer.step([input_query], [response_1], reward)
#     elif user_choice == "2":
#         reward = [torch.tensor(1.0)]
#         train_stats = ppo_trainer.step([input_query], [response_2], reward)
#     else:
#         print("Invalid choice. Skipping model update.")

#     # Repeat the loop for the next query