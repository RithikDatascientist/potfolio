import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, roc_auc_score
)
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier

# Assuming you have your data loaded
# X = train_fe.drop(columns=['id', 'Heart Disease'])
# y = train_fe['Heart Disease']

# Your trained model
best_xgb = XGBClassifier(
    subsample=0.9,
    n_estimators=500,
    min_child_weight=3,
    max_depth=4,
    learning_rate=0.15,
    colsample_bytree=0.8,
    random_state=42
)

# Fit the model
# best_xgb.fit(X, y)

# Get predictions and probabilities for validation (using cross-validation)
# This gives us predictions on training data without overfitting metrics
# y_pred = cross_val_predict(best_xgb, X, y, cv=5, method='predict')
# y_pred_proba = cross_val_predict(best_xgb, X, y, cv=5, method='predict_proba')[:, 1]

# OR if you have a separate validation set:
# y_pred = best_xgb.predict(X_val)
# y_pred_proba = best_xgb.predict_proba(X_val)[:, 1]


# ============================================================================
# 1. CONFUSION MATRIX HEATMAP
# ============================================================================
def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix as a heatmap
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with counts and percentages
    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            annotations.append(
                f"{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)"
            )
    
    annotations = np.array(annotations).reshape(cm.shape)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['No Disease (0)', 'Has Disease (1)'],
        y=['No Disease (0)', 'Has Disease (1)'],
        colorscale='Blues',
        text=annotations,
        texttemplate='%{text}',
        textfont={"size": 14},
        hoverongaps=False,
        hovertemplate='True Label: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=500,
        font=dict(size=12)
    )
    
    return fig


# ============================================================================
# 2. ROC CURVE
# ============================================================================
def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot ROC curve with AUC score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    
    # Diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash'),
        showlegend=True
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=600,
        legend=dict(x=0.6, y=0.1),
        font=dict(size=12)
    )
    
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig


# ============================================================================
# 3. PRECISION-RECALL CURVE
# ============================================================================
def plot_precision_recall_curve(y_true, y_pred_proba):
    """
    Plot Precision-Recall curve
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold_idx = np.argmax(f1_scores[:-1])  # Exclude last point
    
    fig = go.Figure()
    
    # PR curve
    fig.add_trace(go.Scatter(
        x=recall, 
        y=precision,
        mode='lines',
        name='Precision-Recall curve',
        line=dict(color='blue', width=2),
        hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
    ))
    
    # Best F1 point
    fig.add_trace(go.Scatter(
        x=[recall[best_threshold_idx]], 
        y=[precision[best_threshold_idx]],
        mode='markers',
        name=f'Best F1 (threshold={thresholds[best_threshold_idx]:.3f})',
        marker=dict(color='red', size=12, symbol='star'),
        hovertemplate=f'Best F1 Score: {f1_scores[best_threshold_idx]:.3f}<br>' +
                     f'Recall: {recall[best_threshold_idx]:.3f}<br>' +
                     f'Precision: {precision[best_threshold_idx]:.3f}<extra></extra>'
    ))
    
    # Baseline (proportion of positive class)
    baseline = np.sum(y_true) / len(y_true)
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray",
                  annotation_text=f"Baseline: {baseline:.3f}")
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=700,
        height=600,
        legend=dict(x=0.6, y=0.9),
        font=dict(size=12)
    )
    
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig


# ============================================================================
# 4. CLASSIFICATION METRICS BAR CHART
# ============================================================================
def plot_classification_metrics(y_true, y_pred):
    """
    Plot precision, recall, F1-score as bar chart
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Extract metrics for each class
    classes = ['No Disease (0)', 'Has Disease (1)']
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for i, class_name in enumerate(classes):
        class_key = str(i)
        values = [
            report[class_key]['precision'],
            report[class_key]['recall'],
            report[class_key]['f1-score']
        ]
        
        fig.add_trace(go.Bar(
            name=class_name,
            x=metrics,
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
            hovertemplate='%{x}: %{y:.3f}<extra></extra>'
        ))
    
    # Add accuracy line
    accuracy = report['accuracy']
    fig.add_hline(y=accuracy, line_dash="dash", line_color="green",
                  annotation_text=f"Accuracy: {accuracy:.3f}")
    
    fig.update_layout(
        title='Classification Metrics by Class',
        xaxis_title='Metric',
        yaxis_title='Score',
        barmode='group',
        width=800,
        height=500,
        legend=dict(x=0.7, y=0.95),
        font=dict(size=12),
        yaxis=dict(range=[0, 1])
    )
    
    return fig


# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================
def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plot top N most important features
    """
    importance = model.feature_importances_
    
    # Create dataframe and sort
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(top_n)
    
    fig = go.Figure(go.Bar(
        x=feat_imp_df['Importance'],
        y=feat_imp_df['Feature'],
        orientation='h',
        text=feat_imp_df['Importance'].round(4),
        textposition='auto',
        marker=dict(
            color=feat_imp_df['Importance'],
            colorscale='Viridis',
            showscale=True
        ),
        hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Most Important Features',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        width=900,
        height=600,
        font=dict(size=11),
        yaxis=dict(autorange="reversed")  # Most important at top
    )
    
    return fig


# ============================================================================
# 6. THRESHOLD OPTIMIZATION PLOT
# ============================================================================
def plot_threshold_optimization(y_true, y_pred_proba):
    """
    Plot how different metrics change with classification threshold
    """
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    for threshold in thresholds:
        y_pred_temp = (y_pred_proba >= threshold).astype(int)
        
        # Avoid division by zero
        if len(np.unique(y_pred_temp)) > 1:
            report = classification_report(y_true, y_pred_temp, output_dict=True, zero_division=0)
            precisions.append(report['1']['precision'])
            recalls.append(report['1']['recall'])
            f1_scores.append(report['1']['f1-score'])
            accuracies.append(report['accuracy'])
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
            accuracies.append(np.mean(y_true == y_pred_temp))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode='lines', 
                             name='Precision', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls, mode='lines', 
                             name='Recall', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode='lines', 
                             name='F1-Score', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=thresholds, y=accuracies, mode='lines', 
                             name='Accuracy', line=dict(color='purple', dash='dash')))
    
    # Mark best F1 threshold
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    
    fig.add_vline(x=best_threshold, line_dash="dash", line_color="orange",
                  annotation_text=f"Best F1 threshold: {best_threshold:.3f}")
    
    fig.update_layout(
        title='Metrics vs Classification Threshold',
        xaxis_title='Classification Threshold',
        yaxis_title='Score',
        width=900,
        height=600,
        legend=dict(x=0.7, y=0.5),
        font=dict(size=12),
        yaxis=dict(range=[0, 1])
    )
    
    return fig


# ============================================================================
# 7. COMBINED DASHBOARD
# ============================================================================
def plot_metrics_dashboard(y_true, y_pred, y_pred_proba, model, feature_names):
    """
    Create a comprehensive dashboard with multiple metrics
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Confusion Matrix', 'ROC Curve', 
                       'Precision-Recall Curve', 'Classification Metrics'),
        specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annotations = [[f"{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)" 
                    for j in range(len(cm[0]))] for i in range(len(cm))]
    
    fig.add_trace(
        go.Heatmap(z=cm, text=annotations, texttemplate='%{text}',
                   colorscale='Blues', showscale=False),
        row=1, col=1
    )
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode='lines', 
                   name=f'ROC (AUC={roc_auc:.3f})',
                   line=dict(color='darkorange')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                   line=dict(dash='dash', color='navy'),
                   showlegend=False),
        row=1, col=2
    )
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    fig.add_trace(
        go.Scatter(x=recall, y=precision, mode='lines',
                   name='PR Curve', line=dict(color='blue')),
        row=2, col=1
    )
    
    # 4. Classification Metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    for i, class_name in enumerate(['No Disease', 'Has Disease']):
        values = [report[str(i)]['precision'], 
                 report[str(i)]['recall'],
                 report[str(i)]['f1-score']]
        
        fig.add_trace(
            go.Bar(name=class_name, x=metrics, y=values,
                   text=[f'{v:.3f}' for v in values],
                   textposition='auto'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=900,
        width=1400,
        showlegend=True,
        title_text="XGBoost Model Performance Dashboard",
        font=dict(size=10)
    )
    
    return fig


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    """
    Example usage with your XGBoost model
    
    # 1. Train your model and get predictions
    best_xgb.fit(X, y)
    
    # 2. For validation metrics, use cross-validation predictions
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(best_xgb, X, y, cv=5, method='predict')
    y_pred_proba = cross_val_predict(best_xgb, X, y, cv=5, method='predict_proba')[:, 1]
    
    # 3. Generate all plots
    fig1 = plot_confusion_matrix(y, y_pred)
    fig1.show()
    
    fig2 = plot_roc_curve(y, y_pred_proba)
    fig2.show()
    
    fig3 = plot_precision_recall_curve(y, y_pred_proba)
    fig3.show()
    
    fig4 = plot_classification_metrics(y, y_pred)
    fig4.show()
    
    fig5 = plot_feature_importance(best_xgb, X.columns, top_n=15)
    fig5.show()
    
    fig6 = plot_threshold_optimization(y, y_pred_proba)
    fig6.show()
    
    # 4. Or create a comprehensive dashboard
    fig_dashboard = plot_metrics_dashboard(y, y_pred, y_pred_proba, best_xgb, X.columns)
    fig_dashboard.show()
    
    # 5. Save plots as HTML
    fig1.write_html('confusion_matrix.html')
    fig2.write_html('roc_curve.html')
    # etc.
    """
    
    print("Visualization functions loaded successfully!")
    print("\nAvailable functions:")
    print("1. plot_confusion_matrix(y_true, y_pred)")
    print("2. plot_roc_curve(y_true, y_pred_proba)")
    print("3. plot_precision_recall_curve(y_true, y_pred_proba)")
    print("4. plot_classification_metrics(y_true, y_pred)")
    print("5. plot_feature_importance(model, feature_names, top_n=15)")
    print("6. plot_threshold_optimization(y_true, y_pred_proba)")
    print("7. plot_metrics_dashboard(y_true, y_pred, y_pred_proba, model, feature_names)")
    print("\nSee usage example in the script for implementation details.")