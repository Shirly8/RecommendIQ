from sklearn.metrics import mean_squared_error
import torch
from Eval_DataLoader import EvalDataset
from torch.utils.data import DataLoader


def evaluate(model, test_loader, k = 10):
    model.eval()
    model_predictions = []
    actual_predictions = []

    with torch.no_grad():
        for data in test_loader:
            userData, LSTMFeature, categorytensor, target = data
            
            result = model(userData, LSTMFeature, categorytensor)
        
            model_predictions.extend(result.cpu().numpy())
            actual_predictions.extend(target.cpu().numpy)

    rmse = mean_squared_error(actual_predictions, model_predictions, squared=False)
    print(f'RMSE: {rmse}')
    return rmse

if __name__ == '__main__':
    
    csv_file = 'eval_data.csv'
    eval_dataset = EvalDataset(csv_file)
    test_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)


    model = torch.load('ncf_model.pth')  # Load your NCF model
    evaluate(model, test_loader)



