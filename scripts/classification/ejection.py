import torch

def get_validation_rankings(model, validation_data, feature_method, device):

    validation_rankings = [] # 2D array, every neuron by ordered list of output on validation data per neuron

    with torch.no_grad():
        v=0
        for item in validation_data:

            feature_vector = feature_method.predict(item[0].to(device))
            logits, _ = model(feature_vector)

            neuron_outputs = logits.data.tolist()[0] #logits

            # initialize array if we haven't yet
            if len(validation_rankings) == 0:
                for output in neuron_outputs:
                    validation_rankings.append([0.0] * len(validation_data))

            n=0
            for output in neuron_outputs:
                validation_rankings[n][v] = output
                n += 1

            v += 1

    # Rank-order the validation scores
    v=0
    for validation in validation_rankings:
        validation.sort()
        validation_rankings[v] = validation
        v += 1

    return validation_rankings


def get_rank(value, rankings):
    index = 0 # default: ranking = 0

    for ranked_number in rankings:
        if value < ranked_number:
            break #NB: this O(N) loop could be optimized to O(log(N))
        index += 1

    if(index >= len(rankings)):
        index = len(rankings) # maximum: ranking = 1

    elif(index > 0):
        # get linear interpolation between the two closest indexes

        diff = rankings[index] - rankings[index - 1]
        perc = value - rankings[index - 1]
        linear = perc / diff
        index = float(index - 1) + linear

    absolute_ranking = index / len(rankings)

    return(absolute_ranking)



def get_model_outliers(model, unlabeled_data, validation_data, feature_method, device, number=5):

    validation_rankings = get_validation_rankings(model, validation_data, feature_method, device)

    outliers = {}

    with torch.no_grad():
        for item in unlabeled_data:
            feature_vector = feature_method.predict(item[0].to(device))
            logits, _ = model(feature_vector)

            neuron_outputs = logits.data.tolist()[0] #logits

            n=0
            ranks = []
            for output in neuron_outputs:
                rank = get_rank(output, validation_rankings[n])
                ranks.append(rank)
                n += 1

            outliers[item[2]] = 1 - (sum(ranks) / len(neuron_outputs)) # average rank

    out = [k[0] for k, v in sorted(outliers.items(), key=lambda item: item[1], reverse=True)]

    return out[:number]
