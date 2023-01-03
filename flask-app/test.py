import evaluate
rouge = evaluate.load('rouge')

references = [["Where are you from?","Where is he from?"]]
predictions = ["Where is you from?"]
results = rouge.compute(predictions=predictions,references=references)

print("[Score] ", results)