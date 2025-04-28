num_epochs = 15
learning_rate = 25e-7
batch_size = 1
import gc
from unidecode import unidecode
from pylatexenc.latex2text import LatexNodes2Text
from tqdm import tqdm
import random
random.seed(777)
import traceback
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from decimal import Decimal
import re
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, Eq, solve, N
def order_keys(data):
    keys = list(data.keys())
    check = False
    while not check:
        check = True
        for i in range(len(keys)-1):
            for j in range(i+1, len(keys), 1):
                if keys[i].find(keys[j]) >= 0:
                    check = False
                    keys[i], keys[j] = keys[j], keys[i]
    return keys

  
def validate_symbols(equations, allowed_symbols):
    all_symbols = set()
    for eq in equations:
        all_symbols.update(eq.free_symbols)

    invalid_symbols = all_symbols - allowed_symbols
    if invalid_symbols:
        return False
    else:
        return True
def print_and_save(message, filename = "output.txt"):
    print(message)
    
    with open(filename, 'a', encoding = "utf-8") as file:
        file.write(str(message) + '\n')

def solve_system(string_equations, target_var, all_variables):
        vars = {}
        for symbol in all_variables+[target_var]:    
            vars[symbol] = symbols(symbol)
        #print_and_save(vars)
        variable_to_solve = symbols(target_var)
#        print_and_save(variable_to_solve)
        equations = [parse_expr(equation, transformations = "all", local_dict = vars) for equation in string_equations]
#        print_and_save(equations)

        if not validate_symbols(equations, set([vars[key] for key in vars] )):
            return "Error: unknown symbols found"
    
        # Check if the list of equations is empty
        if not equations:
            raise ValueError("The list of equations is empty.")
    
        num_equations = len(equations)
    
        solutions = solve(equations, dict=True)
    
                # Check each solution for a valid numerical result
        for sol in solutions:
            substituted_value = variable_to_solve.subs(sol)
    
                    # Check if the substituted value is a number
            if substituted_value.is_number:
                        return N(substituted_value)
        raise ValueError("No valid numerical solution found")

def math_replace(text, test = True):
    if test:
        return LatexNodes2Text().latex_to_text(text)
        replacements = {
            '×': '*',          # Multiplication sign
            '÷': '/',          # Division sign
            '±': r'\pm',       # Plus-minus sign
            '∓': r'\mp',       # Minus-plus sign
            '√': r'\sqrt{',    # Square root (open brace)
            '∛': r'\sqrt[3]{', # Cube root (open brace)
            '∜': r'\sqrt[4]{', # Fourth root (open brace)
            '²': '**2',        # Superscript 2
            '³': '**3',        # Superscript 3
            'ⁿ': '**n',        # Superscript n
            '∑': r'\sum',      # Summation
            '∏': r'\prod',     # Product
            '∫': r'\int',      # Integral
            '∮': r'\oint',     # Contour integral
            '∞': r'\infty',    # Infinity
            '≠': '!=',         # Not equal to
            '≈': r'\approx',   # Approximately equal to
            '≡': r'\equiv',    # Identical to
            '≤': r'\leq',      # Less than or equal to
            '≥': r'\geq',      # Greater than or equal to
            '≪': r'\ll',       # Much less than
            '≫': r'\gg',       # Much greater than
            '∈': r'\in',       # Element of
            '∉': r'\notin',    # Not an element of
            '∋': r'\ni',       # Contains as member
            '∌': r'\notni',    # Does not contain as member
            '⊂': r'\subset',   # Subset of
            '⊃': r'\supset',   # Superset of
            '⊆': r'\subseteq', # Subset of or equal to
            '⊇': r'\supseteq', # Superset of or equal to
            '∅': r'\emptyset', # Empty set
            '∧': r'\land',     # Logical and
            '∨': r'\lor',      # Logical or
            '¬': r'\neg',      # Logical negation
            '→': r'\to',       # Right arrow
            '←': r'\leftarrow', # Left arrow
            '↔': r'\leftrightarrow', # Left-right arrow
            '∴': r'\therefore', # Therefore
            '∵': r'\because',   # Because
            '∝': r'\propto',    # Proportional to
            '∠': r'\angle',     # Angle
            '∥': r'\parallel',   # Parallel to
            '⊥': r'\perp',      # Perpendicular to
            '∘': r'\circ',      # Circle (composition)
            '∗': '*',           # Asterisk operator
            '⟨': r'\langle',    # Left angle bracket
            '⟩': r'\rangle',    # Right angle bracket
            '⟦': r'\llbracket', # Left double bracket
            '⟧': r'\rrbracket', # Right double bracket
            '⨀': r'\odot',      # Circled dot
            '⨁': r'\oplus',     # Circled plus
            '⨂': r'\otimes',    # Circled times
            '⨄': r'\oplus',     # Circled plus
            '⨆': r'\bigvee',    # Big union
            '⨇': r'\bigcap',    # Big intersection
            '⨈': r'\bigcup',    # Big cup
            '⨉': r'\bigotimes',  # Big product
            '⨊': r'\bigsqcup',   # Big disjoint union
            '⨋': r'\bigoplus',   # Big direct sum
        }
        
        # Replace square root and other roots with parameters
        for key, value in replacements.items():
            text = text.replace(key, value)
        
        # Handle square roots specifically to add closing braces
        text = text.replace(r'\sqrt{', r'\sqrt{')  # This is just to ensure the opening brace is there
        text = text.replace('}', '}')  # Ensure we close the braces properly
    
        return text
def convert_to_decimal(value):
        """Convert a string value to a Decimal, handling fractions and mixed numbers."""
        try:
        # Check for mixed numbers (e.g., "1 1/2")
            if ' ' in value:
                whole, fraction_part = value.split()
                whole = Decimal(whole)
                numerator, denominator = map(Decimal, fraction_part.split('/'))
                return whole + (numerator / denominator)
        # Check for simple fractions (e.g., "1/2")
            elif '/' in value:
                numerator, denominator = map(Decimal, value.split('/'))
                return numerator / denominator
            else:
                return Decimal(value)  # Regular number
        except Exception:
            return None  # Return None if conversion fails

def check_answer(num1, num2, epsilon = .2):
#    print_and_save(num1)
#    print_and_save(num2)
    root1 =  convert_to_decimal(str(num1))
    root2 = convert_to_decimal(str(num2))
#    print_and_save(root1)
#    print_and_save(root2)
    if root1 is None or root2 is None:
        return False
    return abs(root1 - root2) < epsilon
def query_model(prompt, model, tokenizer):
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
      raise ValueError("The tokenizer does not have an EOS token ID.")

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens = 4096,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    generated_ids = outputs[0]

    eos_index = (generated_ids == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_index) > 0:
        generated_ids = generated_ids[:eos_index[0] + 1]

    decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    decoded_prompt = tokenizer.decode(inputs[0], skip_special_tokens=True)

    if decoded_output.startswith(decoded_prompt):
        response = decoded_output[len(decoded_prompt):]
    else:
        response = decoded_output

    return response
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=128, lora_alpha=256, lora_dropout=0.1)
#model_name = "Qwen/Qwen2.5-14B"
model_name = "Qwen/Qwen2.5-0.5B"
model  = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto", quantization_config=bnb_config)
model = get_peft_model(model, peft_config)

tokenizer = AutoTokenizer.from_pretrained(model_name,  padding_side="left", add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
with open("dataset.json", "r", encoding="utf-8") as file:
        data = json.load(file)
random.shuffle(data)
test_data = data[-100:]
validate_data = data[-200:-100]
dataset = [tokenizer(entry["prompt"], padding=False, truncation=False, return_tensors = "pt", return_attention_mask=True) for entry in tqdm(data)]
for i in tqdm(range(len(dataset))):
    dataset[i]["input_ids"] = torch.cat((dataset[i]["input_ids"], torch.tensor([[tokenizer.eos_token_id]])), dim = 1)
    dataset[i]["attention_mask"] = torch.cat((dataset[i]["attention_mask"], torch.tensor([[1]])), dim = 1)
    labels = dataset[i]["input_ids"].clone()  
#    labels[:, :-1] = dataset[i]["input_ids"][:, 1:]  
#    labels[:, -1] = tokenizer.pad_token_id
    dataset[i]["labels"] = labels

train_dataset = dataset[:-200]
test_dataset = dataset[-100:]
def collate_fn(batch):
    input_ids = torch.cat([torch.tensor(item['input_ids']) for item in batch], dim = 0)
    attention_masks = torch.cat([torch.tensor(item['attention_mask']) for item in batch], dim = 0)
    labels = torch.cat([torch.tensor(item['labels']) for item in batch], dim = 0)

    return {
        'input_ids': input_ids.to("cuda"),
        'attention_mask': attention_masks.to("cuda"),
        'labels': labels.to("cuda"),
    }

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn= collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn= collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
max_accuracy = 0
print_and_save("Starting Training")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0  
    num_batches = 0  
    for batch in tqdm(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        epoch_loss += loss.item()  
        num_batches += 1  
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_epoch_loss = epoch_loss / num_batches
    print_and_save(f"Epoch: {epoch+1}, Average training Loss: {avg_epoch_loss}")

    model.eval()  
    test_loss = 0.0
    num_batches = 0

    with torch.no_grad():  
        for batch in tqdm(test_dataloader):
            outputs = model(**batch)
            loss = outputs.loss

            test_loss += loss.item()
            num_batches += 1

    avg_test_loss = test_loss / num_batches
    print_and_save(f"Average Test Loss: {avg_test_loss}")
    correct = 0
    total = 0
    for index, entry in tqdm(enumerate(test_data)):
        total+=1
        question = entry["question"]
        answer = entry["result"]
        prompt = f"### Question:\n{question}\n"
        response = query_model(prompt, model, tokenizer)
#        print_and_save(f"{index}: {response}")

        try:
            chunks = ["###" + part for part in response.split("###") if part]  
            unicode_equations = [math_replace(line.strip()) for line in chunks[2].split("\n")[1:] if len(line.strip())>0]
            all_vars = [var.strip() for var in chunks[3].split("\n")[1].strip().split(",")]
            target_vars = [var.strip() for var in chunks[4].split("\n")[1].strip().split(",")]

            var_dict = {}
            i = 0
            for var in all_vars+target_vars:
                    var_dict[var] = chr(i+65)
                    i+=1
            target_vars = [var_dict[var] for var in target_vars]
#            print_and_save(target_vars)
            ordered_vars = order_keys(var_dict)
            ordered_vars.reverse()
#            print_and_save(ordered_vars)
            for key in ordered_vars:
                for i, equation in enumerate(unicode_equations):
                    unicode_equations[i] = equation.replace(key, var_dict[key])
            unicode_equations = [equation.replace("·", "*").replace('−', '-').replace('×', '*').replace('÷', '/').replace('²', '**2').replace('³', '**3').replace('⁴', '**4').replace('½', '1/2').replace('¼', '1/4').replace('¾', '3/4').replace('π', 'pi').replace('≤', '<=').replace('≥', '>=').replace('√', 'sqrt(').replace('∛', 'cbrt(') for equation in unicode_equations]
            if len(target_vars) == 1:
                    generated_answer = str(solve_system(unicode_equations, target_vars[0], [var_dict[var] for var in var_dict]))
            else: 
                generated_answer = "Too many targets"
            if generated_answer != "Too many targets" and check_answer(answer, generated_answer):
#                print_and_save("correct")
                correct +=1
            else:
                pass
#                print_and_save("Incorrect")
        except Exception :
            pass
#            print_and_save(traceback.format_exc())
    print_and_save(f"Test Accuracy: {correct/total}")
    if correct/total > max_accuracy:
        print_and_save("Saving model...")
        max_accuracy = correct/total
        model.save_pretrained("./peft_model")
        print_and_save("Model saved.")

del model
gc.collect()
torch.cuda.empty_cache()
model  = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto", quantization_config=bnb_config)
model = PeftModel.from_pretrained(model, "./peft_model/")

correct_model = 0
correct_base = 0
base_model_answer = "123"
total = 0
for index, entry in tqdm(enumerate(validate_data)):
        total+=1
        question = entry["question"]
        answer = entry["result"]
        if check_answer(answer, base_model_answer):
            correct_base += 1
        prompt = f"### Question:\n{question}\n"
        response = query_model(prompt, model, tokenizer)
#        print_and_save(f"{index}: {response}")

        try:
            chunks = ["###" + part for part in response.split("###") if part]  
            unicode_equations = [math_replace(line.strip()) for line in chunks[2].split("\n")[1:] if len(line.strip())>0]
            all_vars = [var.strip() for var in chunks[3].split("\n")[1].strip().split(",")]
            target_vars = [var.strip() for var in chunks[4].split("\n")[1].strip().split(",")]

            var_dict = {}
            i = 0
            for var in all_vars+target_vars:
                    var_dict[var] = chr(i+65)
                    i+=1
            target_vars = [var_dict[var] for var in target_vars]
#            print_and_save(target_vars)
            ordered_vars = order_keys(var_dict)
            ordered_vars.reverse()
#            print_and_save(ordered_vars)
            for key in ordered_vars:
                for i, equation in enumerate(unicode_equations):
                    unicode_equations[i] = equation.replace(key, var_dict[key])
            unicode_equations = [equation.replace("·", "*").replace('−', '-').replace('×', '*').replace('÷', '/').replace('²', '**2').replace('³', '**3').replace('⁴', '**4').replace('½', '1/2').replace('¼', '1/4').replace('¾', '3/4').replace('π', 'pi').replace('≤', '<=').replace('≥', '>=').replace('√', 'sqrt(').replace('∛', 'cbrt(') for equation in unicode_equations]
            if len(target_vars) == 1:
                    generated_answer = str(solve_system(unicode_equations, target_vars[0], [var_dict[var] for var in var_dict]))
            else: 
                generated_answer = "Too many targets"
            if generated_answer != "Too many targets" and check_answer(answer, generated_answer):
#                print_and_save("correct")
                correct_model +=1
            else:
                pass
#                print_and_save("Incorrect")
        except Exception :
            pass
#            print_and_save(traceback.format_exc())
print_and_save(f"Validation accuracy for final model: {correct_model/total}")
print_and_save(f"Validation accuracy for base model: {correct_base/total}")
