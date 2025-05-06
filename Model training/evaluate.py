import torch
import argparse
import json
import time
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# Arguments for the evaluation script
def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate QA and Base models using PsychoLex")
    parser.add_argument("--base_model_path", type=str, 
                        default="/data/yl7622/emotion_detection/models--meta-llama--Llama-2-7b-hf", 
                        help="LLaMA base model path")
    parser.add_argument("--lora_path", type=str, 
                        default="/data/yl7622/emotion_detection/qa_lora_checkpoints/qa_lora_final", 
                        help="LoRA weights path for QA model")
    parser.add_argument("--device", type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu", 
                        help="Inference device")
    parser.add_argument("--max_length", type=int, 
                        default=512, 
                        help="Maximum length for generated text")
    parser.add_argument("--temperature", type=float, 
                        default=0.7, 
                        help="Generation temperature")
    parser.add_argument("--top_p", type=float, 
                        default=0.9, 
                        help="Top-p sampling truncation")
    parser.add_argument("--psycholex_model", type=str,
                        default="aminabbasi/PsychoLexLLaMA-8B",
                        help="PsychoLex model ID")
    parser.add_argument("--num_questions", type=int,
                        default=50,
                        help="Number of depression-related questions to generate")
    parser.add_argument("--eval_iterations", type=int,
                        default=1,
                        help="Number of evaluation iterations per question")
    parser.add_argument("--output_file", type=str,
                        default="model_evaluation_results.json",
                        help="File to save evaluation results")
    return parser.parse_args()

# QA Model Class
class QAModel:
    def __init__(self, base_model_path, lora_path, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        print("Loading QA model...")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=True if torch.cuda.is_available() else False
        )
        
        # Load LoRA weights
        print(f"Loading LoRA weights from: {lora_path}")
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        
        # Set model to evaluation mode
        self.model.eval()
        print("QA model loaded successfully")
    
    def generate_response(self, question, max_length=512, temperature=0.7, top_p=0.9):
        # Format input text
        input_text = f"Question: {question}\nAnswer:"
        
        # Encode text
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and process output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part (remove question part)
        answer = generated_text.split("Answer:")[-1].strip()
        return answer

# Base Model Class
class LlamaBaseModel:
    def __init__(self, model_path, device="cuda:0", use_8bit=False):
        """Initialize Llama base model"""
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print("Loading base model...")
        if use_8bit and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                load_in_8bit=True
            )
        else:
            # Set lower precision to save memory
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )
        
        # Set model to evaluation mode
        self.model.eval()
        print("Base model loaded successfully")
    
    def generate_response(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """Generate response using the model"""
        # Format input text for consistency with QA model
        input_text = f"Question: {prompt}\nAnswer:"
        
        # Encode input text
        inputs = self.tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
        else:
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part for consistency with QA model
        answer = generated_text.split("Answer:")[-1].strip()
        return answer

# ... existing code ...

class PsychoLexEvaluator:
    def __init__(self, model_id, device="cuda:0"):
        print(f"Loading PsychoLex model from: {model_id}")
        model_kwargs = {"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {}
        device_map = "auto" if torch.cuda.is_available() else None
        
        self.pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs=model_kwargs,
            device_map=device_map
        )
        print("PsychoLex model loaded successfully")
    
    def generate_questions(self, num_questions=50):
        """Generate depression-related questions using PsychoLex"""
        print(f"Generating {num_questions} depression-related questions...")
        
        # Use a direct, simple prompt to generate questions
        prompt = f"Regard yourself as a teacher teaching about depression detecting, and you are trying to test how well your students know these knowledge. Generate {num_questions} specific questions about psychology assessment about depression to check if the students know how to assess depression or so. Number them as 1., 2., 3., etc."
        
        try:
            # Generate the text directly without using chat format
            output = self.pipeline(
                prompt,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7
            )
            
            # Extract the generated text 
            generated_text = output[0]["generated_text"]
            
            # Print raw output for debugging
            print("Raw model output sample (first 200 chars):")
            print(generated_text[:200] + "...")
            
            # Extract questions using regex
            import re
            
            # Look for numbered patterns like "1. What are the symptoms of depression?"
            questions = []
            pattern = r'(?:^|\n)\s*(\d+)[.:\)]\s*(.*?)(?=(?:\n\s*\d+[.:\)]|\Z))'
            matches = re.findall(pattern, generated_text, re.DOTALL)
            
            for _, question_text in matches:
                question = question_text.strip()
                if question and len(question) > 10:  # Minimal validation
                    questions.append(question)
            
            print(f"Extracted {len(questions)} questions using numbered pattern")
            
            # If that didn't work, try another pattern for "Question N:"
            if len(questions) < 5:
                pattern = r'(?:Question|Q)[\s:]*((?:\d+|[A-Za-z]+))[\s:]+(.+?)(?=(?:Question|Q)[\s:]*(?:\d+|[A-Za-z]+)[\s:]|$)'
                matches = re.findall(pattern, generated_text, re.DOTALL)
                questions = []
                for _, question_text in matches:
                    question = question_text.strip()
                    if question and len(question) > 10:
                        questions.append(question)
                
                print(f"Extracted {len(questions)} questions using 'Question N:' pattern")
            
            # If still not enough questions, try line by line for sentences ending with "?"
            if len(questions) < 5:
                questions = []
                for line in generated_text.split('\n'):
                    line = line.strip()
                    # Remove any numbering at the beginning
                    line = re.sub(r'^\s*\d+[.:\)]?\s*', '', line)
                    if '?' in line and len(line) > 15:
                        questions.append(line)
                
                print(f"Extracted {len(questions)} questions using question mark pattern")
        
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            questions = []
        
        # If we still don't have enough questions, use hardcoded questions
        if len(questions) < num_questions:
            print("Using default depression-related questions")
            default_questions = [
                "What are the main symptoms of depression?",
                "How is depression diagnosed by healthcare professionals?",
                "What treatments are most effective for severe depression?",
                "How does depression affect daily functioning and quality of life?",
                "What are the known risk factors for developing depression?",
                "How is depression different from normal sadness or grief?",
                "Can depression be cured completely, or is it managed long-term?",
                "What role do antidepressant medications play in treating depression?",
                "How can family members best support someone with depression?",
                "What lifestyle changes can help manage depression symptoms?",
                "Is depression hereditary or primarily caused by environmental factors?",
                "How does depression affect sleep patterns and energy levels?",
                "What is the relationship between anxiety disorders and depression?",
                "How effective is cognitive behavioral therapy for treating depression?",
                "Can regular exercise help reduce symptoms of depression?",
                "What are the potential side effects of common depression medications?",
                "How does chronic depression affect brain structure and function?",
                "What are the warning signs of suicidal thoughts in depressed individuals?",
                "How does depression present differently in children versus adults?",
                "What is the role of social support in recovery from depression?",
                "How can mindfulness and meditation practices help with depression?",
                "What is treatment-resistant depression and how is it managed?",
                "How do hormonal changes contribute to depression in women?",
                "What is the connection between substance abuse and depression?",
                "How effective are natural or alternative treatments for depression?",
                "What is postpartum depression and how is it treated?",
                "How does depression impact academic or work performance?",
                "What are the physical symptoms associated with depression?",
                "How does seasonal affective disorder differ from other forms of depression?",
                "What is the recommended diet to help manage depression symptoms?",
                "How does depression affect personal relationships and social interactions?",
                "What are the neurochemical changes that occur in the brains of depressed people?",
                "How quickly do antidepressants typically work to relieve symptoms?",
                "What is the relationship between chronic pain and depression?",
                "How can stress management techniques help with depression?",
                "What are the long-term effects of untreated depression?",
                "How does depression impact decision-making abilities?",
                "What is the connection between inflammation and depression?",
                "How does light therapy work for treating certain types of depression?",
                "What are the symptoms of high-functioning depression?",
                "How does depression affect memory and concentration?",
                "What is the relationship between depression and weight changes?",
                "How can digital health technologies help monitor and manage depression?",
                "What is the role of genetic testing in depression treatment?",
                "How does depression differ across different cultures and societies?",
                "What is the impact of depression on the immune system?",
                "How can art therapy or music therapy help with depression?",
                "What coping mechanisms are most effective for managing depression symptoms?",
                "How does depression affect motivation and goal-setting?",
                "What is the connection between depression and feelings of guilt or shame?"
            ]
            
            # Fill in with default questions
            remaining_count = num_questions - len(questions)
            questions.extend(default_questions[:remaining_count])
        
        # Ensure we have exactly the requested number of questions
        if len(questions) > num_questions:
            questions = questions[:num_questions]
        
        # Print a sample of the questions
        print(f"\nSample of generated questions:")
        for i, q in enumerate(questions[:5]):
            print(f"  {i+1}. {q}")
        if len(questions) > 5:
            print(f"  ... and {len(questions)-5} more questions")
        
        return questions
    
    def evaluate_response(self, question, response):
        """Evaluate a model's response to a depression-related question"""
        try:
            # Create a simple, direct prompt for evaluation
            eval_prompt = f"""
                Rate the following response to a question about depression on a scale from 0-10:

                QUESTION: {question}

                RESPONSE: {response}

                Evaluate based on:
                - Accuracy of information
                - Sensitivity to mental health issues
                - Helpfulness of the response
                - Appropriate tone

                Rate from 0 (completely inappropriate) to 10 (excellent in all aspects).
                Your rating (just the number):"""
            
            # Get the evaluation
            output = self.pipeline(
                eval_prompt,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1  # Very low temperature for consistent scoring
            )
            
            # Extract the generated text
            generated_text = output[0]["generated_text"]
            
            # Print raw output for debugging (only for the first few evaluations)
            if hasattr(self, 'eval_count'):
                self.eval_count += 1
            else:
                self.eval_count = 1
                
            if self.eval_count <= 3:
                print("\nRaw evaluation output sample:")
                print(generated_text[-100:])  # Show the end where the rating should be
            
            # Try to extract a numerical rating
            # First, extract everything after our prompt
            response_part = generated_text.replace(eval_prompt, "").strip()
            
            # Look for a number at the beginning of the response
            import re
            rating_match = re.search(r'^(\d+(?:\.\d+)?)', response_part)
            if rating_match:
                try:
                    rating = float(rating_match.group(1))
                    # Ensure rating is in valid range
                    rating = max(0, min(10, rating))
                    return rating
                except:
                    pass
            
            # If that doesn't work, look for any number from 0-10 in the text
            number_matches = re.findall(r'\b(10|[0-9](?:\.\d+)?)\b', response_part)
            for num in number_matches:
                try:
                    value = float(num)
                    if 0 <= value <= 10:
                        return value
                except:
                    continue
            
            # Default score if extraction fails
            return 5.0
            
        except Exception as e:
            print(f"Error evaluating response: {str(e)}")
            return 5.0  # Default score if evaluation fails

def evaluate_models(args):
    # Initialize models
    psycholex = PsychoLexEvaluator(args.psycholex_model)
    qa_model = QAModel(args.base_model_path, args.lora_path, args.device)
    base_model = LlamaBaseModel(args.base_model_path, args.device)
    
    # Generate questions
    questions = psycholex.generate_questions(args.num_questions)
    
    # Create data structure to store results
    results = {
        "questions": questions,
        "qa_model_scores": [],
        "base_model_scores": [],
        "qa_wins": 0,
        "base_wins": 0,
        "ties": 0,
        "detailed_results": []
    }
    
    # Evaluate each question
    for i, question in enumerate(tqdm(questions, desc="Evaluating questions")):
        print(f"\nEvaluating Question {i+1}/{len(questions)}: {question}")
        
        qa_max_score = 0
        base_max_score = 0
        qa_best_response = ""
        base_best_response = ""
        
        question_results = {
            "question": question,
            "qa_iterations": [],
            "base_iterations": [],
            "qa_max_score": 0,
            "base_max_score": 0
        }
        
        # Evaluate multiple times and take the best score
        for iteration in range(args.eval_iterations):
            print(f"  Iteration {iteration+1}/{args.eval_iterations}")
            
            # Get responses from both models
            qa_response = qa_model.generate_response(
                question, 
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            base_response = base_model.generate_response(
                question, 
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            # Evaluate responses
            qa_score = psycholex.evaluate_response(question, qa_response)
            base_score = psycholex.evaluate_response(question, base_response)
            
            # Record iteration results
            question_results["qa_iterations"].append({
                "response": qa_response,
                "score": qa_score
            })
            
            question_results["base_iterations"].append({
                "response": base_response,
                "score": base_score
            })
            
            # Update best scores
            if qa_score > qa_max_score:
                qa_max_score = qa_score
                qa_best_response = qa_response
            
            if base_score > base_max_score:
                base_max_score = base_score
                base_best_response = base_response
            
            print(f"  QA Model Score: {qa_score}, Base Model Score: {base_score}")
        
        # Record best scores
        results["qa_model_scores"].append(qa_max_score)
        results["base_model_scores"].append(base_max_score)
        question_results["qa_max_score"] = qa_max_score
        question_results["base_max_score"] = base_max_score
        question_results["qa_best_response"] = qa_best_response
        question_results["base_best_response"] = base_best_response
        
        # Determine winner
        if qa_max_score > base_max_score:
            results["qa_wins"] += 1
            question_results["winner"] = "qa_model"
        elif base_max_score > qa_max_score:
            results["base_wins"] += 1
            question_results["winner"] = "base_model"
        else:
            results["ties"] += 1
            question_results["winner"] = "tie"
        
        results["detailed_results"].append(question_results)
        
        print(f"Best QA Score: {qa_max_score}, Best Base Score: {base_max_score}")
        print(f"Current standings - QA wins: {results['qa_wins']}, Base wins: {results['base_wins']}, Ties: {results['ties']}")
    
    # Calculate average scores
    results["qa_avg_score"] = np.mean(results["qa_model_scores"])
    results["base_avg_score"] = np.mean(results["base_model_scores"])
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n===== EVALUATION SUMMARY =====")
    print(f"QA Model Average Score: {results['qa_avg_score']:.2f}")
    print(f"Base Model Average Score: {results['base_avg_score']:.2f}")
    print(f"QA Model wins: {results['qa_wins']}/{args.num_questions} questions")
    print(f"Base Model wins: {results['base_wins']}/{args.num_questions} questions")
    print(f"Ties: {results['ties']}/{args.num_questions} questions")
    print(f"Detailed results saved to: {args.output_file}")
    
    return results

def main():
    args = parse_arguments()
    start_time = time.time()
    results = evaluate_models(args)
    end_time = time.time()
    
    print(f"\nTotal evaluation time: {(end_time - start_time) / 60:.2f} minutes")
    
    # Determine overall winner
    print("\n===== FINAL VERDICT =====")
    if results["qa_avg_score"] > results["base_avg_score"]:
        print(f"QA Model WINS with score {results['qa_avg_score']:.2f} vs Base Model {results['base_avg_score']:.2f}")
    elif results["base_avg_score"] > results["qa_avg_score"]:
        print(f"Base Model WINS with score {results['base_avg_score']:.2f} vs QA Model {results['qa_avg_score']:.2f}")
    else:
        print(f"IT'S A TIE! Both models scored {results['qa_avg_score']:.2f}")

if __name__ == "__main__":
    main()