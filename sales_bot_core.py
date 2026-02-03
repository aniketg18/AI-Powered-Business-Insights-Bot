# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 12:48:03 2026

@author: Aniket
This is the core logic file for the Sales AI Bot.
It contains:
- Data loading and preprocessing
- Business insights calculations
- GPT-2 model loading and rewriting
- Question routing function
"""
# ==============================
# IMPORTS
# ==============================
import pandas as pd 
import torch 
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ==============================
# DATA LOAD AND PREPROCESSING
# ==============================
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# Convert date columns
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])


# Total profit by region
profit_by_region = df.groupby("Region")["Profit"].sum().sort_values(ascending=False)

# Profit margin by category
category_summary = df.groupby("Category")[["Sales", "Profit"]].sum()
category_summary["Profit Margin"] = category_summary["Profit"] / category_summary["Sales"]

# -----------------------------
# Functions for total profit and total sales
# -----------------------------
def total_profit_text():
    """
    Returns total profit across all regions as text.
    """
    total_profit = df["Profit"].sum()
    return f"The total profit across all regions is {total_profit:,.0f}."

def total_sales_text():
    """
    Returns total sales across all regions as text.
    """
    total_sales = df["Sales"].sum()
    return f"The total sales across all regions is {total_sales:,.0f}."

# -----------------------------
# Functions for profit by region and category
# -----------------------------
def get_profit_by_region():
    return (
        df.groupby("Region")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )
print("\nFunction Output - Profit by Region:")
print(get_profit_by_region())

def format_profit_by_region():
    data = get_profit_by_region()
    formatted = data.round(0).astype(int)
    return formatted
print("\nFormatted Profit by Region:")
print(format_profit_by_region())

def get_profit_margin_by_category():
    summary = df.groupby("Category")[["Sales", "Profit"]].sum()
    summary["Profit Margin (%)"] = (summary["Profit"] / summary["Sales"]) * 100
    return summary["Profit Margin (%)"].round(2)
print("\nProfit Margin by Category (%):")
print(get_profit_margin_by_category())


def profit_by_region_text():
    data = format_profit_by_region()
    
    lines = []
    for region, profit in data.items():
        lines.append(f"{region} region generated a profit of {profit:,}.")
    
    return " ".join(lines)
print("\nProfit by Region (Text):")
print(profit_by_region_text())

def profit_margin_text():
    margins = get_profit_margin_by_category()
    
    lines = []
    for category, margin in margins.items():
        lines.append(f"{category} has a profit margin of {margin} percent.")
    
    return " ".join(lines)
print("\nProfit Margin (Text):")
print(profit_margin_text())

# ==============================
# GPT-2 MODEL LOAD AND REWRITE FUNCTION
# ==============================
# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def gpt_rewrite(text):
    prompt = (
        "Rewrite the following business insight in a professional and clear manner:\n\n"
        + text
    )

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1] + 60,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGPT-2 Insight (Profit by Region):")
print(gpt_rewrite(profit_by_region_text()))


def gpt_rewrite(text):
    prompt = (
        "Rewrite the following business insight in a professional and clear manner:\n\n"
        + text
    )

    # Encode the input and create attention mask
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)

    # Generate GPT-2 output
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,    # avoids warning
        max_length=inputs.shape[1] + 100, # increase generation length
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=2.0,           # avoids repeating phrases
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGPT-2 Insight (Profit by Region):")
print(gpt_rewrite(profit_by_region_text()))

# ==============================
# QUESTION ROUTING FUNCTION
# ==============================
def route_question(user_question):
    question = user_question.lower()

    if "profit by region" in question or "region profit" in question:
        return gpt_rewrite(profit_by_region_text())

    elif "profit margin" in question or "margin by category" in question:
        return gpt_rewrite(profit_margin_text())

    elif "total profit" in question:
        return gpt_rewrite(total_profit_text())

    elif "total sales" in question or "sales count" in question or "how many sales" in question:
        return gpt_rewrite(total_sales_text())

    else:
        return "Sorry, I don't have that insight yet. Please ask about profit, margin, or sales."

# ==============================
# OPTIONAL TESTING (REMOVE OR COMMENT IN PRODUCTION)
# ==============================
questions = [
    "Show me profit by region",
    "What is the profit margin for categories?",
    "Tell me total profit",
    "How many sales were there?"
]

for q in questions:
    print(f"\nQuestion: {q}")
    print("GPT-2 Response:")
    print(route_question(q))
    

    
