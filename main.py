from model_development import *

def main():
    
    y, x = preprocess()
    x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test = split_impute_scale_pipe(x=x, y=y, test_size=0.15)

    data_dict = {
        'x_train': x_train_scaled, 
        'x_val' : x_val_scaled, 
        'x_test': x_test_scaled, 
        'y_train': y_train, 
        'y_val': y_val, 
        'y_test': y_test
    }

    focal_base = instantiate_model().tfLogisticRegression(activation_function = 'sigmoid' )

    focal_base.compile(optimizer=keras.optimizers.Adam(0.0005), loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.675, gamma=4))

    trained_focal_model_history = fit_model().train_model(model = focal_base, data_dict=data_dict)

    pred, pred_prob  = fit_model().test_model(trained_model = focal_base, x_test = data_dict['x_test'])

    metrics = model_metrics().classification_metrics(y_actual = data_dict['y_test'], y_pred = pred, y_prob = pred_prob)

    focal_base.save('/Users/ryan/Desktop/ResearchRepo/HSCRPmlResearch/venv/focal_model.h5')

if __name__ ==  '__main__':
    main()
