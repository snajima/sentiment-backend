############################################################################################
# Choosing Weights for the Three Algorithms

# Labels to output for user
label_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

#Temporary Filler for Algorithms 1-3
def alg1_run():
    return random.rand(28)
def alg2_run():
    return random.rand(28)
def alg3_run():
    return random.rand(28)

# Initialize weights of all three algorithms to 1/3
alg1_w = alg2_w = alg3_w = 1/3
top_w = 0.5
# Initialize weights of all three algorithms to 1/3
def update_weights(top_w):
    # Run all three algorithms
    alg1_res = alg1_run()
    alg2_res = alg2_run()
    alg3_res = alg3_run()
    # Multiply the results of each algorithm by the weights and add everything together
    # Truncate to the top three results
    alg_cum = alg1_w * alg1_res + alg2_w * alg2_res + alg3_w * alg3_res
    top_three = sorted(alg_cum, key= lambda i: i[1], reverse=True)[0:3]
    # Store User's sentence and output as (x,y)
    # Need help here from @Shungo
    top_labels = [label_cols[top_three[0][0]], label_cols[top_three[1][0]]], label_cols[top_three[2][0]]
    choice = input(f"Choose between: {top_labels[0]}, {top_labels[1]}, and {top_labels[2]}")
    # Give user the ability to pick one of three, let us decide, or do custom input:
    if choice == '':
        return alg1_w, alg2_w, alg3_w
    # After user decides between the top three, change the weights of the algorithm
    # Initialize hyperparameters x that represents the weight of the top emotion (x < 1)
    # For the algorithm that had the highest amount of the top emotion, increase weight of the algorithm by its (weight of the top "emotion"
    # - average of the emotion of the three algorithms) multiplied by x and (1-weight)
    # (i.e. if 0.5 of happy came from this algorithm, after first iteration, with total happy of 0.9, 
    # increase weight by ((0.5-0.3) * 2/3 * x)
    if choice in top_labels:
        choice_index = label_cols.index(choice)
        choice_alg1 = alg1_res[choice_index]
        choice_alg2 = alg2_res[choice_index]
        choice_alg3 = alg3_res[choice_index]
        choice_avg = sum(choice_alg1, choice_alg2, choice_alg3)/3
        choice_val1 = (choice_val1 - choice_avg) * top_w * alg1_w * (1 - alg1_w)
        choice_val2 = (choice_val2 - choice_avg) * top_w * alg2_w * (1 - alg2_w)
        choice_val3 = (choice_val3 - choice_avg) * top_w * alg3_w * (1 - alg3_w)
        choice_val_sum = choice_val1 + choice_val2 + choice_val3
        # Do the same for the two other algorithms
        # Reweigh all algorithms on a scale of 0 to 1 by dividing by sum and multiplying by 3
        choice_val1 = choice_val1 / choice_val_sum
        choice_val2 = choice_val2 / choice_val_sum
        choice_val3 = choice_val3  / choice_val_sum
        return choice_val1, choice_val2, choice_val3

    # Create variables representing the proportion of weight relative to each 

    # Potentially decrease top_w after a certain number of iterations

    # If frequent miscategorization, potentially switch to scheme where weights of algorithms are dependent on previous predictions of top emotions
    # (i.e. if one category usually predicts sad, and the user constantly says we are predicting wrong, then we switch to a higher weight of sad predictor)
    # Potentially decrease x, y once there is more information?

#############################################################################
# Algorithm 2