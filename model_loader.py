import os
import pickle
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

class ModelLoader:
    """
    Class untuk memuat model
    """
    
    def __init__(self, model_folder_path):
        self.model_folder_path = model_folder_path
        self.neural_model = None
        self.scaler = None
        self.features = None
        self.feature_stats = None
        self.habit_rules = None
        self.metadata = None
        self.target_names = ["study_hours_per_day", "sleep_hours", "social_media_hours", "netflix_hours"]
        
    def load_all_components(self):
        """
        Muat semua komponen model
        """
        try:
            print(f"Memuat model dari: {self.model_folder_path}")
            
            # 1. Load neural network model
            print("Memuat neural network model...")
            self._load_neural_model()
            
            # 2. Load selected features
            print("Memuat daftar fitur...")
            self._load_features()
            
            # 3. Load scaler
            print("Memuat scaler...")
            self._load_scaler()
            
            # 4. Load feature statistics
            print("Memuat statistik fitur...")
            self._load_feature_stats()
            
            # 5. Load habit rules
            print("Memuat habit rules...")
            self._load_habit_rules()
            
            # 6. Load metadata
            print("Memuat metadata...")
            self._load_metadata()
            
            print("SEMUA KOMPONEN BERHASIL DIMUAT!")
            return True
            
        except Exception as e:
            print(f"Error saat memuat model: {str(e)}")
            return False
    
    def _load_neural_model(self):
        """Load neural network model dengan berbagai format"""
        try:
            import tensorflow as tf
        except ImportError:
            print("TensorFlow tidak terinstall, skip neural model")
            return
        
        model_loaded = False
        
        # Coba format .keras
        model_path_keras = os.path.join(self.model_folder_path, "neural_network_model.keras")
        if os.path.exists(model_path_keras):
            try:
                self.neural_model = tf.keras.models.load_model(model_path_keras)
                print("Neural model (.keras) berhasil dimuat")
                model_loaded = True
            except Exception as e:
                print(f".keras gagal: {str(e)}")
        
        # Coba format .h5
        if not model_loaded:
            model_path_h5 = os.path.join(self.model_folder_path, "neural_network_model.h5")
            if os.path.exists(model_path_h5):
                try:
                    self.neural_model = tf.keras.models.load_model(model_path_h5)
                    print("Neural model (.h5) berhasil dimuat")
                    model_loaded = True
                except Exception as e:
                    print(f".h5 gagal: {str(e)}")
        
        # Coba SavedModel
        if not model_loaded:
            model_path_saved = os.path.join(self.model_folder_path, "neural_network_savedmodel")
            if os.path.exists(model_path_saved):
                try:
                    self.neural_model = tf.keras.models.load_model(model_path_saved)
                    print("Neural model (SavedModel) berhasil dimuat")
                    model_loaded = True
                except Exception as e:
                    print(f"SavedModel gagal: {str(e)}")
        
        # Coba load dari weights + architecture
        if not model_loaded:
            weights_path = os.path.join(self.model_folder_path, "model_weights.h5")
            arch_path = os.path.join(self.model_folder_path, "model_architecture.json")
            
            if os.path.exists(weights_path) and os.path.exists(arch_path):
                try:
                    with open(arch_path, 'r') as f:
                        model_json = f.read()
                    
                    self.neural_model = tf.keras.models.model_from_json(model_json)
                    self.neural_model.load_weights(weights_path)
                    print("Neural model (weights+arch) berhasil dimuat")
                    model_loaded = True
                except Exception as e:
                    print(f"Weights+arch gagal: {str(e)}")
        
        if not model_loaded:
            print("Neural model tidak dapat dimuat, akan menggunakan rule-based saja")
    
    def _load_features(self):
        """Load daftar fitur"""
        features_path = os.path.join(self.model_folder_path, "selected_features.json")
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.features = json.load(f)
            print(f"Fitur berhasil dimuat: {len(self.features)} fitur")
        else:
            print("File features tidak ditemukan")
    
    def _load_scaler(self):
        """Load scaler dengan fallback"""
        # Coba load scaler pickle
        scaler_path = os.path.join(self.model_folder_path, "scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Scaler (pickle) berhasil dimuat")
                return
            except Exception as e:
                print(f"Scaler pickle gagal: {str(e)}")
        
        # Fallback ke scaler info
        scaler_info_path = os.path.join(self.model_folder_path, "scaler_info.json")
        if os.path.exists(scaler_info_path):
            try:
                with open(scaler_info_path, 'r') as f:
                    scaler_info = json.load(f)
                
                # Buat scaler manual dari info
                self.scaler = StandardScaler()
                if self.features:
                    self.scaler.mean_ = np.array([scaler_info['feature_means'].get(f, 0) for f in self.features])
                    self.scaler.scale_ = np.array([scaler_info['feature_stds'].get(f, 1) for f in self.features])
                print("Scaler (dari info) berhasil dimuat")
                return
            except Exception as e:
                print(f"Scaler info gagal: {str(e)}")
        
        print("Scaler tidak dapat dimuat, akan menggunakan normalisasi manual")
    
    def _load_feature_stats(self):
        """Load statistik fitur"""
        feature_stats_path = os.path.join(self.model_folder_path, "feature_stats.json")
        if os.path.exists(feature_stats_path):
            with open(feature_stats_path, 'r') as f:
                self.feature_stats = json.load(f)
            print("Statistik fitur berhasil dimuat")
        else:
            print("Statistik fitur tidak ditemukan")
    
    def _load_habit_rules(self):
        """Load habit rules"""
        rules_path = os.path.join(self.model_folder_path, "habit_rules.json")
        if os.path.exists(rules_path):
            with open(rules_path, 'r') as f:
                self.habit_rules = json.load(f)
            print("Habit rules berhasil dimuat")
        else:
            print("Habit rules tidak ditemukan")
    
    def _load_metadata(self):
        """Load metadata"""
        metadata_path = os.path.join(self.model_folder_path, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print("Metadata berhasil dimuat")
        else:
            print("Metadata tidak ditemukan")
    
    def preprocess_input(self, user_input):
        """
        Preprocess input
        """
        try:
            if not self.features:
                return None
                
            # Pastikan semua fitur ada dalam input
            processed_input = {}
            for feature in self.features:
                if feature in user_input:
                    processed_input[feature] = float(user_input[feature])
                else:
                    # Gunakan nilai default
                    if self.feature_stats and feature in self.feature_stats:
                        processed_input[feature] = self.feature_stats[feature]['mean']
                    else:
                        processed_input[feature] = 0.0
            
            # Convert ke array
            input_array = np.array([list(processed_input.values())])
            
            # Apply scaler jika ada
            if self.scaler is not None:
                try:
                    input_array = self.scaler.transform(input_array)
                except Exception as e:
                    print(f"Scaler transform gagal: {str(e)}, menggunakan normalisasi manual")
                    # Normalisasi manual menggunakan feature stats
                    if self.feature_stats:
                        for i, feature in enumerate(self.features):
                            if feature in self.feature_stats:
                                mean = self.feature_stats[feature]['mean']
                                std = self.feature_stats[feature]['std']
                                input_array[0][i] = (input_array[0][i] - mean) / (std + 1e-8)
            
            return input_array
            
        except Exception as e:
            print(f"Error preprocessing: {str(e)}")
            return None
    
    def predict_with_neural_model(self, user_input):
        """
        Prediksi menggunakan neural network dengan bounds checking
        """
        try:
            if self.neural_model is None:
                return None
            
            # Preprocess input
            processed_input = self.preprocess_input(user_input)
            if processed_input is None:
                return None
            
            # Prediksi
            predictions = self.neural_model.predict(processed_input, verbose=0)
            
            # Convert ke dictionary
            result = {}
            for i, target in enumerate(self.target_names):
                if i < predictions.shape[1]:
                    result[target] = float(predictions[0][i])
            
            # BOUNDS CHECKING - Pastikan nilai realistis
            result['study_hours_per_day'] = max(1, min(12, result.get('study_hours_per_day', 4)))
            result['sleep_hours'] = max(6, min(10, result.get('sleep_hours', 7.5)))
            result['social_media_hours'] = max(0.5, min(6, result.get('social_media_hours', 2)))
            result['netflix_hours'] = max(0.5, min(4, result.get('netflix_hours', 1.5)))
            
            # Validasi total waktu tidak melebihi 24 jam
            total_hours = (result['study_hours_per_day'] + result['sleep_hours'] + 
                          result['social_media_hours'] + result['netflix_hours'])
            
            if total_hours > 20:  # Sisakan 4 jam untuk aktivitas lain
                # Scale down proportional untuk non-sleep activities
                non_sleep_total = (result['study_hours_per_day'] + 
                                  result['social_media_hours'] + result['netflix_hours'])
                available_non_sleep = 20 - result['sleep_hours']
                
                if non_sleep_total > available_non_sleep and non_sleep_total > 0:
                    scale_factor = available_non_sleep / non_sleep_total
                    result['study_hours_per_day'] *= scale_factor
                    result['social_media_hours'] *= scale_factor
                    result['netflix_hours'] *= scale_factor
            
            # Round to 1 decimal place
            for key in result:
                result[key] = round(result[key], 1)
            
            return result
            
        except Exception as e:
            print(f"Error neural prediction: {str(e)}")
            return None
    
    def predict_with_rules(self, user_input):
        """
        Prediksi
        """
        try:
            if self.habit_rules is None:
                # Buat rules default jika tidak ada
                self.habit_rules = self._create_default_rules()
            
            predictions = {}
            
            for habit_name, rules in self.habit_rules.items():
                base_value = rules['base']
                adjustment = 0
                
                for feature, params in rules['factors'].items():
                    if feature in user_input:
                        user_value = user_input[feature]
                        optimal_value = params['optimal']
                        weight = params['weight']
                        
                        if optimal_value != 0:
                            deviation = (user_value - optimal_value) / optimal_value
                        else:
                            deviation = user_value - optimal_value
                        
                        adjustment += weight * deviation
                
                # Apply adjustment dengan bounds
                predicted_value = base_value + adjustment
                
                # Apply realistic bounds
                if habit_name == 'study_hours_per_day':
                    predicted_value = max(1, min(12, predicted_value))
                elif habit_name == 'sleep_hours':
                    predicted_value = max(6, min(10, predicted_value))
                elif habit_name in ['social_media_hours', 'netflix_hours']:
                    predicted_value = max(0.5, min(4, predicted_value))
                
                predictions[habit_name] = round(predicted_value, 1)
            
            return predictions
            
        except Exception as e:
            print(f"Error rule-based prediction: {str(e)}")
            return None
    
    def _create_default_rules(self):
        """
        Buat rules default jika tidak ada
        """
        return {
            'study_hours_per_day': {
                'base': 4.0,
                'factors': {
                    'exam_score': {'weight': 0.05, 'optimal': 80},
                    'attendance_percentage': {'weight': 0.03, 'optimal': 90},
                    'mental_health_rating': {'weight': 0.2, 'optimal': 8},
                    'academic_lifestyle': {'weight': 0.3, 'optimal': 2}
                }
            },
            'sleep_hours': {
                'base': 7.5,
                'factors': {
                    'mental_health_rating': {'weight': 0.2, 'optimal': 8},
                    'exercise_frequency': {'weight': 0.1, 'optimal': 3},
                    'age': {'weight': -0.1, 'optimal': 20}
                }
            },
            'social_media_hours': {
                'base': 2.0,
                'factors': {
                    'total_screen_time': {'weight': 0.4, 'optimal': 3},
                    'mental_health_rating': {'weight': -0.1, 'optimal': 8},
                    'academic_lifestyle': {'weight': -0.2, 'optimal': 2}
                }
            },
            'netflix_hours': {
                'base': 1.5,
                'factors': {
                    'total_screen_time': {'weight': 0.3, 'optimal': 3},
                    'study_efficiency': {'weight': -0.05, 'optimal': 20},
                    'mental_health_rating': {'weight': 0.1, 'optimal': 7}
                }
            }
        }
    
    def predict(self, user_input, use_neural=True):
        """
        Prediksi utama dengan fallback
        """
        # Coba neural model dulu
        if use_neural and self.neural_model is not None:
            neural_result = self.predict_with_neural_model(user_input)
            if neural_result is not None:
                return neural_result, "neural_network"
        
        # Fallback ke rule-based
        rule_result = self.predict_with_rules(user_input)
        if rule_result is not None:
            return rule_result, "rule_based"
        
        return None, "error"
    
    def generate_recommendations(self, predictions):
        """
        Generate rekomendasi berdasarkan prediksi
        """
        recommendations = {}
        
        # Study hours recommendation
        study_hours = predictions.get('study_hours_per_day', 4.0)
        if study_hours < 3:
            study_advice = "Tingkatkan waktu belajar secara bertahap untuk hasil akademik yang lebih baik"
        elif study_hours > 8:
            study_advice = "Pertimbangkan untuk mengoptimalkan efisiensi belajar"
        else:
            study_advice = "Waktu belajar sudah dalam range optimal"
        
        recommendations['study_hours'] = {
            'recommended_hours': round(study_hours, 1),
            'advice': study_advice
        }
        
        # Sleep hours recommendation
        sleep_hours = predictions.get('sleep_hours', 7.5)
        if sleep_hours < 7:
            sleep_advice = "Tidur kurang dapat mengganggu konsentrasi. Usahakan 7-8 jam"
        elif sleep_hours > 9:
            sleep_advice = "Tidur berlebihan dapat mengurangi produktivitas harian"
        else:
            sleep_advice = "Durasi tidur sudah optimal"
        
        recommendations['sleep_hours'] = {
            'recommended_hours': round(sleep_hours, 1),
            'advice': sleep_advice
        }
        
        # Social media recommendation
        social_hours = predictions.get('social_media_hours', 2.0)
        if social_hours > 3:
            social_advice = "Batasi penggunaan media sosial untuk menghindari distraksi"
        elif social_hours < 1:
            social_advice = "Sedikit interaksi sosial digital dapat membantu relaksasi"
        else:
            social_advice = "Penggunaan media sosial dalam batas wajar"
        
        recommendations['social_media_hours'] = {
            'recommended_hours': round(social_hours, 1),
            'advice': social_advice
        }
        
        # Entertainment recommendation
        netflix_hours = predictions.get('netflix_hours', 1.5)
        if netflix_hours > 2:
            netflix_advice = "Batasi waktu hiburan untuk menjaga keseimbangan akademik"
        elif netflix_hours < 1:
            netflix_advice = "Sedikit waktu hiburan dapat membantu mengurangi stress"
        else:
            netflix_advice = "Waktu hiburan sudah seimbang"
        
        recommendations['entertainment_hours'] = {
            'recommended_hours': round(netflix_hours, 1),
            'advice': netflix_advice
        }
        
        return recommendations
    
    def get_model_info(self):
        """
        Dapatkan informasi model
        """
        if self.metadata:
            return {
                "model_name": self.metadata.get("model_name", "Unknown"),
                "timestamp": self.metadata.get("timestamp", "Unknown"),
                "features_count": len(self.features) if self.features else 0,
                "features": self.features,
                "target_variables": self.target_names,
                "tensorflow_version": self.metadata.get("tensorflow_version", "Unknown"),
                "architecture": self.metadata.get("model_architecture", "Unknown"),
                "loader_type": "ModelLoader"
            }
        else:
            return {
                "error": "Metadata tidak tersedia",
                "loader_type": "ModelLoader",
                "features": self.features,
                "neural_model_available": self.neural_model is not None
            }

def load_latest_model(save_directory="saved_models"):
    """
    Load model terbaru dari direktori
    """
    if not os.path.exists(save_directory):
        print(f"Direktori {save_directory} tidak ditemukan")
        return None
    
    # Cari folder model terbaru
    model_folders = [f for f in os.listdir(save_directory) 
                    if os.path.isdir(os.path.join(save_directory, f))]
    
    if not model_folders:
        print(f"Tidak ada model ditemukan di {save_directory}")
        return None
    
    # Sort berdasarkan timestamp
    model_folders.sort(reverse=True)
    latest_folder = model_folders[0]
    
    model_path = os.path.join(save_directory, latest_folder)
    print(f"Loading model terbaru: {latest_folder}")
    
    # Load model
    loader = ModelLoader(model_path)
    if loader.load_all_components():
        return loader
    else:
        return None

def test_model_loader():
    """
    Test function untuk model loader
    """
    print("TESTING MODEL LOADER")
    print("=" * 40)
    
    # Load model
    model_loader = load_latest_model()
    
    if model_loader is None:
        print("Tidak dapat load model")
        return False
    
    # Test data
    sample_input = {
        'attendance_percentage': 85,
        'exercise_frequency': 3,
        'internet_quality_encoded': 2,
        'mental_health_rating': 7,
        'exam_score': 90,
        'total_screen_time': 3.5,
        'study_efficiency': 16,
        'sleep_quality_score': 56,
        'academic_lifestyle': 1.5
    }
    
    # Test prediksi
    print("\nTesting prediksi...")
    predictions, method = model_loader.predict(sample_input)
    
    if predictions:
        print(f"Prediksi berhasil menggunakan: {method}")
        print(f"Study hours: {predictions.get('study_hours_per_day', 0):.1f}")
        print(f"Sleep hours: {predictions.get('sleep_hours', 0):.1f}")
        print(f"Social media: {predictions.get('social_media_hours', 0):.1f}")
        print(f"Netflix: {predictions.get('netflix_hours', 0):.1f}")
        
        # Test rekomendasi
        print("\nTesting rekomendasi...")
        recommendations = model_loader.generate_recommendations(predictions)
        for habit_type, data in recommendations.items():
            print(f"   {habit_type}: {data['recommended_hours']} jam - {data['advice']}")
        
        print(f"\nMODEL LOADER BERFUNGSI DENGAN BAIK!")
        return True
    else:
        print("Prediksi gagal")
        return False

if __name__ == "__main__":
    test_model_loader()