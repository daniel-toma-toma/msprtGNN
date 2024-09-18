# for 10 features
upfd4_threshold_dict = {
    "msprtGNN" : 0.5,
    "upfd-sage": 0.8,
    "gcnfn" : 0.7,
    "naive" : 0.998,
    "markovMSPRT" : 0.95,
    "quickstop" : 0.93,
}
upfd3_threshold_dict = {
    "msprtGNN" : 0.72,#0.75,
    "upfd-sage": 0.95,
    "gcnfn" : 0.95,
    "naive" : 0.998,
    "markovMSPRT" : 0.95,
    "quickstop" : 0.93,
}
weibo_threshold_dict = {
    "msprtGNN" : 0.36, #0.42,
    "upfd-sage": 0.4,
    "gcnfn" : 0.55,
    "naive" : 0.9,
    "markovMSPRT" : 0.99,
    "quickstop" : 0.9,
}
threshold_dict = {
    "upfd3" : upfd3_threshold_dict,
    "upfd4" : upfd4_threshold_dict,
    "weibo": weibo_threshold_dict,
    "weibo3": weibo_threshold_dict,
}