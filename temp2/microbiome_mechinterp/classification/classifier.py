import numpy as np
import pandas as pd
import biom
from qiime2 import Artifact, Metadata
from qiime2.plugins.sample_classifier.pipelines import classify_samples, heatmap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import os

class FeatureClassifier:
    """Simple Random Forest classifier for feature evaluation"""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rf = None
        
    def evaluate_features(self, features, labels, n_folds=5):
        """Evaluate features using Random Forest with cross-validation"""
        self.rf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        
        # Cross-validation
        scores = cross_val_score(self.rf, features, labels, cv=n_folds)
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        # Train on all data
        self.rf.fit(features, labels)
        predictions = self.rf.predict(features)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return accuracy, f1, self.rf

class Q2SampleClassifier:
    """QIIME2 Sample Classifier wrapper for IBD prediction with extensive debugging"""
    
    def __init__(self):
        self.results = None
        
    def sparse_features_to_biom_qza(self, sparse_features, sample_ids, feature_prefix="sparse_"):
        """Convert sparse features to QIIME2 artifact with debugging"""
        
        print(f"\n=== DEBUGGING BIOM/QZA CONVERSION ===")
        print(f"Input sparse_features shape: {sparse_features.shape}")
        print(f"Input sample_ids length: {len(sample_ids)}")
        print(f"Feature prefix: {feature_prefix}")
        
        # Create feature IDs
        feature_ids = [f"{feature_prefix}{i}" for i in range(sparse_features.shape[1])]
        print(f"Created {len(feature_ids)} feature IDs")
        print(f"First 5 feature IDs: {feature_ids[:5]}")
        print(f"First 5 sample IDs: {sample_ids[:5]}")
        
        # Debug sparse features statistics
        print(f"\nSparse features statistics:")
        print(f"  Min value: {sparse_features.min():.6f}")
        print(f"  Max value: {sparse_features.max():.6f}")
        print(f"  Mean: {sparse_features.mean():.6f}")
        print(f"  Std: {sparse_features.std():.6f}")
        print(f"  Zero values: {(sparse_features == 0).sum()} / {sparse_features.size}")
        print(f"  NaN values: {np.isnan(sparse_features).sum()}")
        print(f"  Inf values: {np.isinf(sparse_features).sum()}")
        
        # Create BIOM table
        print(f"\nCreating BIOM table...")
        print(f"Transposing features for BIOM: {sparse_features.T.shape}")
        
        biom_table = biom.Table(sparse_features.T, feature_ids, sample_ids)
        print(f"BIOM table created successfully")
        print(f"BIOM table shape: {biom_table.shape}")
        print(f"BIOM table feature count: {biom_table.length(axis='observation')}")
        print(f"BIOM table sample count: {biom_table.length(axis='sample')}")
        
        # Convert to DataFrame for inspection
        biom_table_df = biom_table.to_dataframe().T
        print(f"\nBIOM table as DataFrame:")
        print(f"DataFrame shape: {biom_table_df.shape}")
        print(f"DataFrame columns (first 5): {list(biom_table_df.columns[:5])}")
        print(f"DataFrame index (first 5): {list(biom_table_df.index[:5])}")
        print(f"DataFrame preview:")
        print(biom_table_df.iloc[:3, :3])
        
        # Convert to QIIME2 artifact
        print(f"\nConverting to QIIME2 artifact...")
        qza_artifact = Artifact.import_data('FeatureTable[Frequency]', biom_table)
        print(f"QZA artifact created: {qza_artifact}")
        print(f"QZA artifact type: {qza_artifact.type}")
        print(f"QZA artifact UUID: {qza_artifact.uuid}")
        
        # View QZA as DataFrame
        qza_artifact_df = qza_artifact.view(pd.DataFrame)
        print(f"\nQZA artifact as DataFrame:")
        print(f"QZA DataFrame shape: {qza_artifact_df.shape}")
        print(f"QZA DataFrame preview:")
        print(qza_artifact_df.iloc[:3, :3])
        
        print(f"=== BIOM/QZA CONVERSION COMPLETE ===\n")
        
        return qza_artifact, biom_table
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, label_names, approach_name="features"):
        """Train and evaluate using QIIME2 sample classifier with extensive debugging"""
        
        print(f"\n{'='*60}")
        print(f"DEBUGGING Q2 SAMPLE CLASSIFIER - {approach_name.upper()}")
        print(f"{'='*60}")
        
        print(f"Input data shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        print(f"  label_names: {label_names}")
        
        print(f"\nLabel distribution:")
        print(f"  y_train unique: {np.unique(y_train, return_counts=True)}")
        print(f"  y_test unique: {np.unique(y_test, return_counts=True)}")
        
        print(f"Converting {approach_name} features to QIIME2 format...")
        
        # Combine train and test data for QIIME2 (it handles the split internally)
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.hstack([y_train, y_test])
        
        print(f"\nCombined data:")
        print(f"  X_combined shape: {X_combined.shape}")
        print(f"  y_combined shape: {y_combined.shape}")
        print(f"  y_combined unique: {np.unique(y_combined, return_counts=True)}")
        
        # Create sample IDs
        train_samples = [f"train_sample_{i}" for i in range(len(X_train))]
        test_samples = [f"test_sample_{i}" for i in range(len(X_test))]
        all_sample_ids = train_samples + test_samples
        
        print(f"\nSample IDs:")
        print(f"  Total samples: {len(all_sample_ids)}")
        print(f"  First 5 sample IDs: {all_sample_ids[:5]}")
        print(f"  Last 5 sample IDs: {all_sample_ids[-5:]}")
        
        # Convert sparse features to BIOM/QZA format
        qza_table, biom_table = self.sparse_features_to_biom_qza(
            X_combined, all_sample_ids, f"{approach_name}_sparse_"
        )
        
        # Create metadata DataFrame
        print(f"\nCreating metadata DataFrame...")
        metadata_df = pd.DataFrame({
            'sample-id': all_sample_ids,
            'diagnosis': [label_names[label] for label in y_combined]
        })
        print(f"Metadata DataFrame before setting index:")
        print(f"  Shape: {metadata_df.shape}")
        print(f"  Columns: {list(metadata_df.columns)}")
        print(f"  First 5 rows:")
        print(metadata_df.head())
        print(f"  Diagnosis value counts:")
        print(metadata_df['diagnosis'].value_counts())
        
        metadata_df.set_index('sample-id', inplace=True)
        print(f"\nMetadata DataFrame after setting index:")
        print(f"  Shape: {metadata_df.shape}")
        print(f"  Index name: {metadata_df.index.name}")
        print(f"  First 5 index values: {list(metadata_df.index[:5])}")
        print(f"  Sample of metadata:")
        print(metadata_df.head())
        
        # Convert to QIIME2 Metadata
        print(f"\nConverting to QIIME2 Metadata...")
        try:
            q2_metadata = Metadata(metadata_df)
            print(f"QIIME2 Metadata created successfully")
            print(f"  Metadata columns: {q2_metadata.columns}")
            
            diagnosis_column = q2_metadata.get_column('diagnosis')
            print(f"Diagnosis column extracted:")
            print(f"  Column type: {type(diagnosis_column)}")
            print(f"  Column name: {diagnosis_column.name}")
            print(f"  Sample diagnosis column:")
            diagnosis_series = diagnosis_column.to_series()
            print(f"  Series shape: {diagnosis_series.shape}")
            print(f"  Series value counts:")
            print(diagnosis_series.value_counts())
            print(f"  First 5 diagnosis values: {list(diagnosis_series.head())}")
            
        except Exception as e:
            print(f"ERROR creating QIIME2 metadata: {str(e)}")
            raise e
        
        print(f"\nRunning QIIME2 sample classifier for {approach_name}...")
        print(f"Classification parameters:")
        print(f"  test_size: 0.2")
        print(f"  cv: 5")
        print(f"  random_state: 42")
        print(f"  n_estimators: 100")
        print(f"  estimator: GradientBoostingClassifier")
        print(f"  optimize_feature_selection: True")
        print(f"  parameter_tuning: True")
        
        try:
            # Run QIIME2 classify_samples
            results = classify_samples(
                table=qza_table,
                metadata=diagnosis_column,
                test_size=0.2,
                step=0.1,
                cv=5,
                random_state=42,
                n_estimators=100,
                estimator='GradientBoostingClassifier',
                optimize_feature_selection=True,
                parameter_tuning=True,
                missing_samples='ignore'
            )
            
            print(f"\n✓ Classification completed successfully!")
            print(f"Results type: {type(results)}")
            print(f"Results attributes: {dir(results)}")
            
            # Examine each result component
            print(f"\nExamining result components:")
            try:
                accuracy = results.accuracy_results
                print(f"  ✓ Accuracy results: {type(accuracy)} - {accuracy.uuid}")
            except Exception as e:
                print(f"  ✗ Accuracy error: {e}")
                
            try:
                predictions = results.predictions
                print(f"  ✓ Predictions: {type(predictions)} - {predictions.uuid}")
                predictions_df = predictions.view(pd.DataFrame)
                print(f"    Predictions DataFrame shape: {predictions_df.shape}")
                print(f"    Predictions DataFrame columns: {list(predictions_df.columns)}")
                print(f"    Predictions preview:")
                print(predictions_df.head())
                print(f"    Prediction value counts:")
                print(predictions_df.iloc[:, 0].value_counts() if predictions_df.shape[1] > 0 else "No columns")
            except Exception as e:
                print(f"  ✗ Predictions error: {e}")
                
            try:
                feature_importance = results.feature_importance
                print(f"  ✓ Feature importance: {type(feature_importance)} - {feature_importance.uuid}")
                importance_df = feature_importance.view(pd.DataFrame)
                print(f"    Feature importance DataFrame shape: {importance_df.shape}")
                print(f"    Feature importance DataFrame columns: {list(importance_df.columns)}")
                print(f"    Top 5 important features:")
                print(importance_df.head())
            except Exception as e:
                print(f"  ✗ Feature importance error: {e}")
                
            try:
                sample_estimator = results.sample_estimator
                print(f"  ✓ Sample estimator: {type(sample_estimator)} - {sample_estimator.uuid}")
            except Exception as e:
                print(f"  ✗ Sample estimator error: {e}")
            
            try:
                probabilities = results.probabilities
                print(f"  ✓ Probabilities: {type(probabilities)} - {probabilities.uuid}")
                prob_df = probabilities.view(pd.DataFrame)
                print(f"    Probabilities DataFrame shape: {prob_df.shape}")
                print(f"    Probabilities DataFrame columns: {list(prob_df.columns)}")
                print(f"    Probabilities preview:")
                print(prob_df.head())
            except Exception as e:
                print(f"  ✗ Probabilities error: {e}")
            
            # Save results with debugging
            print(f"\nSaving results files...")
            # Create qiime2_results directory if it doesn't exist
            os.makedirs('qiime2_results', exist_ok=True)
            
            try:
                results.accuracy_results.save(f'qiime2_results/q2_accuracy_{approach_name}.qzv')
                print(f"  ✓ Saved: qiime2_results/q2_accuracy_{approach_name}.qzv")
            except Exception as e:
                print(f"  ✗ Error saving accuracy: {e}")
                
            try:
                results.predictions.save(f'qiime2_results/q2_predictions_{approach_name}.qzv')
                print(f"  ✓ Saved: qiime2_results/q2_predictions_{approach_name}.qzv")
            except Exception as e:
                print(f"  ✗ Error saving predictions: {e}")
                
            try:
                results.feature_importance.save(f'qiime2_results/q2_feature_importance_{approach_name}.qza')
                print(f"  ✓ Saved: qiime2_results/q2_feature_importance_{approach_name}.qza")
            except Exception as e:
                print(f"  ✗ Error saving feature importance: {e}")
            
            # Create simplified results dictionary for compatibility
            results_dict = {
                'GradientBoostingClassifier': {
                    'model': results.sample_estimator,
                    'accuracy': 'See q2_accuracy_*.qzv file',
                    'f1_micro': 'See q2_accuracy_*.qzv file', 
                    'f1_macro': 'See q2_accuracy_*.qzv file',
                    'f1_weighted': 'See q2_accuracy_*.qzv file',
                    'predictions': results.predictions,
                    'qiime2_results': results
                }
            }
            
            print(f"\nResults dictionary created successfully!")
            print(f"Keys: {list(results_dict.keys())}")
            print(f"GBC keys: {list(results_dict['GradientBoostingClassifier'].keys())}")
            
            print(f"{'='*60}")
            print(f"Q2 SAMPLE CLASSIFIER DEBUG COMPLETE - SUCCESS")
            print(f"{'='*60}\n")
            
            return results_dict
            
        except Exception as e:
            print(f"\n❌ ERROR in QIIME2 classification: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback:")
            traceback.print_exc()
            
            return {
                'GradientBoostingClassifier': {
                    'model': None,
                    'accuracy': 0.0,
                    'f1_micro': 0.0,
                    'f1_macro': 0.0, 
                    'f1_weighted': 0.0,
                    'predictions': None,
                    'error': str(e)
                }
            }
    
    def create_heatmap(self, qza_table, feature_importance, sample_metadata, approach_name="features"):
        """Create QIIME2 heatmap visualization with debugging"""
        
        print(f"\n=== DEBUGGING HEATMAP CREATION - {approach_name.UPPER()} ===")
        print(f"Input parameters:")
        print(f"  qza_table type: {type(qza_table)}")
        print(f"  feature_importance type: {type(feature_importance)}")
        print(f"  sample_metadata type: {type(sample_metadata)}")
        print(f"  approach_name: {approach_name}")
        
        try:
            print(f"\nExamining QZA table...")
            qza_df = qza_table.view(pd.DataFrame)
            print(f"  QZA table shape: {qza_df.shape}")
            print(f"  QZA table columns (first 5): {list(qza_df.columns[:5])}")
            print(f"  QZA table index (first 5): {list(qza_df.index[:5])}")
            
            print(f"\nExamining feature importance...")
            importance_df = feature_importance.view(pd.DataFrame)
            print(f"  Feature importance shape: {importance_df.shape}")
            print(f"  Feature importance columns: {list(importance_df.columns)}")
            print(f"  Top 5 important features:")
            print(importance_df.head())
            
            print(f"\nExamining sample metadata...")
            metadata_series = sample_metadata.to_series()
            print(f"  Metadata series shape: {metadata_series.shape}")
            print(f"  Metadata value counts:")
            print(metadata_series.value_counts())
            
            print(f"\nCreating heatmap for {approach_name}...")
            
            heatmap_results = heatmap(
                table=qza_table,
                importance=feature_importance,
                sample_metadata=sample_metadata,
                feature_count=50,
                importance_threshold=0.00,
                group_samples=True,
                normalize=True,
                metric='euclidean',
                method='average',
                cluster='features',
                color_scheme='RdBu_r'
            )
            
            print(f"  ✓ Heatmap created successfully!")
            print(f"  Heatmap results type: {type(heatmap_results)}")
            print(f"  Heatmap attributes: {dir(heatmap_results)}")
            
            # Save heatmap
            os.makedirs('qiime2_results', exist_ok=True)
            heatmap_results.heatmap.save(f'qiime2_results/q2_heatmap_{approach_name}.qzv')
            print(f"  ✓ Heatmap saved as qiime2_results/q2_heatmap_{approach_name}.qzv")
            print(f"  Heatmap UUID: {heatmap_results.heatmap.uuid}")
            
            print(f"=== HEATMAP CREATION COMPLETE - SUCCESS ===\n")
            
            return heatmap_results
            
        except Exception as e:
            print(f"\n❌ Error creating heatmap: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback:")
            traceback.print_exc()
            print(f"=== HEATMAP CREATION COMPLETE - FAILED ===\n")
            return None

print("Q2SampleClassifier class with extensive debugging defined successfully!")
