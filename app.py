from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
from datetime import datetime
import traceback

# Inisialisasi Flask app
app = Flask(__name__)
CORS(app)

# Global variable untuk model
model_loader = None

# Dummy loader sebagai fallback
class DummyLoader:
    def __init__(self):
        self.features = [
            'age', 'gender_encoded', 'part_time_job_encoded', 
            'attendance_percentage', 'diet_quality_encoded',
            'exercise_frequency', 'parental_education_level_encoded',
            'internet_quality_encoded', 'mental_health_rating',
            'extracurricular_participation_encoded', 'exam_score',
            'total_screen_time', 'study_efficiency', 
            'sleep_quality_score', 'academic_lifestyle'
        ]
        
    def predict(self, user_input, use_neural=True):
        # Rule-based prediction sebagai fallback
        exam_score = user_input.get('exam_score', 75)
        mental_health = user_input.get('mental_health_rating', 7)
        age = user_input.get('age', 20)
        attendance = user_input.get('attendance_percentage', 80)
        
        study_hours = max(2, min(8, 2 + (exam_score - 50) / 25 + (attendance - 70) / 30))
        sleep_hours = max(6, min(9, 8 - (10 - mental_health) * 0.2))
        social_hours = max(1, min(4, 3 - (exam_score - 60) / 40))
        netflix_hours = max(0.5, min(3, 2 - (exam_score - 60) / 50))
        
        predictions = {
            'study_hours_per_day': study_hours,
            'sleep_hours': sleep_hours,
            'social_media_hours': social_hours,
            'netflix_hours': netflix_hours
        }
        return predictions, "rule_based_fallback"
    
    def generate_recommendations(self, predictions):
        recommendations = {}
        
        # Study hours
        study_hours = predictions.get('study_hours_per_day', 4)
        if study_hours < 3:
            study_advice = "Tingkatkan waktu belajar untuk hasil akademik yang lebih baik"
        elif study_hours > 6:
            study_advice = "Pertimbangkan untuk mengoptimalkan efisiensi belajar"
        else:
            study_advice = "Waktu belajar sudah dalam range optimal"
        recommendations['study_hours'] = {
            'recommended_hours': round(study_hours, 1),
            'advice': study_advice
        }
        
        # Sleep hours
        sleep_hours = predictions.get('sleep_hours', 7.5)
        if sleep_hours < 7:
            sleep_advice = "Tidur kurang dapat mengganggu konsentrasi. Usahakan 7-8 jam"
        elif sleep_hours > 8.5:
            sleep_advice = "Tidur berlebihan dapat mengurangi produktivitas harian"
        else:
            sleep_advice = "Durasi tidur sudah optimal"
        recommendations['sleep_hours'] = {
            'recommended_hours': round(sleep_hours, 1),
            'advice': sleep_advice
        }
        
        # Social media
        social_hours = predictions.get('social_media_hours', 2)
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
        
        # Entertainment
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
        return {
            "model_name": "fallback_model",
            "loader_type": "DummyLoader",
            "features": self.features,
            "status": "Menggunakan rule-based fallback"
        }

def fix_predictions(predictions):
    """"""
    if not predictions:
        return predictions
    
    # Bounds checking
    fixed = {}
    fixed['study_hours_per_day'] = max(1, min(12, predictions.get('study_hours_per_day', 4)))
    fixed['sleep_hours'] = max(6, min(10, predictions.get('sleep_hours', 7.5)))
    fixed['social_media_hours'] = max(0.5, min(6, predictions.get('social_media_hours', 2)))
    fixed['netflix_hours'] = max(0.5, min(4, predictions.get('netflix_hours', 1.5)))
    
    # Pastikan total tidak melebihi 20 jam (sisakan 4 jam untuk aktivitas lain)
    total = (fixed['study_hours_per_day'] + fixed['sleep_hours'] + 
             fixed['social_media_hours'] + fixed['netflix_hours'])
    
    if total > 20:
        # Scale down proporsi untuk non-sleep activities
        non_sleep_total = (fixed['study_hours_per_day'] + 
                          fixed['social_media_hours'] + fixed['netflix_hours'])
        available_non_sleep = 20 - fixed['sleep_hours']
        
        if non_sleep_total > available_non_sleep:
            scale = available_non_sleep / non_sleep_total
            fixed['study_hours_per_day'] *= scale
            fixed['social_media_hours'] *= scale
            fixed['netflix_hours'] *= scale
    
    # Round to 1 decimal place
    for key in fixed:
        fixed[key] = round(fixed[key], 1)
    
    return fixed

def initialize_model():
    """Inisialisasi model saat aplikasi startup"""
    global model_loader
    try:
        print("Menginisialisasi model...")
        
        try:
            from model_loader import load_latest_model
            model_loader = load_latest_model()
            if model_loader:
                print("Model berhasil diinisialisasi dengan ModelLoader")
                return True
        except ImportError:
            print("model_loader.py tidak ditemukan")
        except Exception as e:
            print(f"Error loading ModelLoader: {str(e)}")
        
        # Fallback ke dummy loader
        model_loader = DummyLoader()
        print("Menggunakan fallback model (rule-based)")
        return True
        
    except Exception as e:
        print(f"Error inisialisasi model: {str(e)}")
        model_loader = DummyLoader()
        print("Menggunakan fallback model karena error")
        return True

@app.route('/', methods=['GET'])
def home():
    """Endpoint home untuk cek status API"""
    return jsonify({
        "message": "API Prediksi Kebiasaan Mahasiswa",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_loader is not None,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "GET /": "Status API",
            "GET /model/info": "Informasi model",
            "POST /predict": "Prediksi single mahasiswa",
            "POST /predict/batch": "Prediksi batch mahasiswa",
            "POST /validate/input": "Validasi data input",
            "GET /examples": "Contoh data input",
            "GET /health": "Health check"
        }
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Endpoint untuk mendapatkan informasi model"""
    try:
        if model_loader is None:
            return jsonify({
                "error": "Model belum diinisialisasi",
                "status": "error"
            }), 500
        
        info = model_loader.get_model_info()
        return jsonify({
            "status": "success",
            "model_info": info
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint utama untuk prediksi kebiasaan mahasiswa"""
    try:
        if model_loader is None:
            return jsonify({
                "error": "Model belum diinisialisasi",
                "status": "error"
            }), 500
        
        # Ambil data dari request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "Data input tidak valid. Pastikan mengirim JSON data.",
                "status": "error"
            }), 400
        
        # Validasi input dasar
        required_fields = ['age', 'gender_encoded', 'attendance_percentage', 'mental_health_rating', 'exam_score']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Field yang wajib: {missing_fields}",
                "status": "error",
                "hint": "Minimal berikan: age, gender_encoded, attendance_percentage, mental_health_rating, exam_score"
            }), 400
        
        # Set default values untuk field yang hilang
        default_values = {
            'part_time_job_encoded': 0,
            'diet_quality_encoded': 1,
            'exercise_frequency': 2,
            'parental_education_level_encoded': 2,
            'internet_quality_encoded': 1,
            'extracurricular_participation_encoded': 0,
            'total_screen_time': 3.0,
            'study_efficiency': data.get('exam_score', 75) / 5,
            'sleep_quality_score': data.get('mental_health_rating', 7) * 8,
            'academic_lifestyle': 1.0
        }
        
        # Fill missing values
        for field, default_val in default_values.items():
            if field not in data:
                data[field] = default_val
        
        # Prediksi
        predictions, method = model_loader.predict(data, use_neural=True)
        
        # Perbaiki prediksi yang tidak realistis
        predictions = fix_predictions(predictions)
        
        if predictions is None:
            return jsonify({
                "error": "Prediksi gagal. Silakan cek data input.",
                "status": "error"
            }), 500
        
        # Generate rekomendasi
        recommendations = model_loader.generate_recommendations(predictions)
        
        # Hitung insight tambahan
        study_hours = recommendations.get('study_hours', {}).get('recommended_hours', 4)
        social_hours = recommendations.get('social_media_hours', {}).get('recommended_hours', 2)
        entertainment_hours = recommendations.get('entertainment_hours', {}).get('recommended_hours', 1.5)
        
        total_productive = study_hours
        total_leisure = social_hours + entertainment_hours
        
        if (total_productive + total_leisure) > 0:
            productive_ratio = total_productive / (total_productive + total_leisure)
        else:
            productive_ratio = 0
        
        # Status berdasarkan produktivitas
        if productive_ratio > 0.7:
            productivity_status = "Sangat Produktif"
        elif productive_ratio > 0.5:
            productivity_status = "Cukup Seimbang"
        else:
            productivity_status = "Perlu Perbaikan"
        
        # Response
        response = {
            "status": "success",
            "prediction_method": method,
            "predictions": predictions,
            "recommendations": recommendations,
            "insights": {
                "productivity_ratio": round(productive_ratio, 3),
                "productivity_status": productivity_status,
                "total_productive_hours": round(total_productive, 1),
                "total_leisure_hours": round(total_leisure, 1),
                "work_life_balance": "Baik" if 0.4 <= productive_ratio <= 0.7 else "Perlu Penyesuaian"
            },
            "input_data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error dalam prediksi: {error_trace}")
        
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Endpoint untuk prediksi batch (multiple mahasiswa)"""
    try:
        if model_loader is None:
            return jsonify({
                "error": "Model belum diinisialisasi",
                "status": "error"
            }), 500
        
        data = request.get_json()
        
        if not data or 'students' not in data:
            return jsonify({
                "error": "Format data tidak valid. Expected: {'students': [student_data1, student_data2, ...]}",
                "status": "error"
            }), 400
        
        students_data = data['students']
        
        if not isinstance(students_data, list):
            return jsonify({
                "error": "Data students harus berupa list",
                "status": "error"
            }), 400
        
        if len(students_data) > 50:
            return jsonify({
                "error": "Maksimal 50 mahasiswa per batch",
                "status": "error"
            }), 400
        
        results = []
        errors = []
        
        for i, student_data in enumerate(students_data):
            try:
                # Set default values
                default_values = {
                    'part_time_job_encoded': 0,
                    'diet_quality_encoded': 1,
                    'exercise_frequency': 2,
                    'parental_education_level_encoded': 2,
                    'internet_quality_encoded': 1,
                    'extracurricular_participation_encoded': 0,
                    'total_screen_time': 3.0,
                    'study_efficiency': student_data.get('exam_score', 75) / 5,
                    'sleep_quality_score': student_data.get('mental_health_rating', 7) * 8,
                    'academic_lifestyle': 1.0
                }
                
                for field, default_val in default_values.items():
                    if field not in student_data:
                        student_data[field] = default_val
                
                predictions, method = model_loader.predict(student_data, use_neural=True)
                predictions = fix_predictions(predictions)
                
                if predictions is None:
                    errors.append({
                        "student_index": i,
                        "error": "Prediksi gagal untuk student ini"
                    })
                    continue
                
                recommendations = model_loader.generate_recommendations(predictions)
                
                student_result = {
                    "student_index": i,
                    "predictions": predictions,
                    "recommendations": recommendations,
                    "input_data": student_data
                }
                
                results.append(student_result)
                
            except Exception as e:
                errors.append({
                    "student_index": i,
                    "error": str(e)
                })
        
        return jsonify({
            "status": "success",
            "total_students": len(students_data),
            "successful_predictions": len(results),
            "failed_predictions": len(errors),
            "results": results,
            "errors": errors if errors else None,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/validate/input', methods=['POST'])
def validate_input():
    """Endpoint untuk validasi data input sebelum prediksi"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "Data input tidak valid",
                "status": "error"
            }), 400
        
        validation_result = {
            "status": "success",
            "input_valid": True,
            "provided_fields": list(data.keys()),
            "missing_basic_fields": [],
            "invalid_fields": [],
            "field_validation": {}
        }
        
        # Basic required fields
        basic_required = ['age', 'gender_encoded', 'attendance_percentage', 'mental_health_rating', 'exam_score']
        
        # Cek missing basic fields
        for field in basic_required:
            if field not in data:
                validation_result["missing_basic_fields"].append(field)
                validation_result["input_valid"] = False
        
        # Validasi tipe data dan range
        field_validations = {
            'age': {'type': 'number', 'min': 17, 'max': 30},
            'gender_encoded': {'type': 'integer', 'min': 0, 'max': 1},
            'attendance_percentage': {'type': 'number', 'min': 0, 'max': 100},
            'mental_health_rating': {'type': 'number', 'min': 1, 'max': 10},
            'exam_score': {'type': 'number', 'min': 0, 'max': 100}
        }
        
        for field, value in data.items():
            if field in field_validations:
                validation = field_validations[field]
                field_result = {"valid": True, "messages": []}
                
                # Cek tipe data
                try:
                    if validation['type'] == 'integer':
                        value = int(value)
                    else:
                        value = float(value)
                except (ValueError, TypeError):
                    field_result["valid"] = False
                    field_result["messages"].append(f"Harus berupa {validation['type']}")
                    validation_result["input_valid"] = False
                    validation_result["invalid_fields"].append(field)
                    continue
                
                # Cek range
                if 'min' in validation and value < validation['min']:
                    field_result["valid"] = False
                    field_result["messages"].append(f"Nilai minimum: {validation['min']}")
                    validation_result["input_valid"] = False
                    if field not in validation_result["invalid_fields"]:
                        validation_result["invalid_fields"].append(field)
                
                if 'max' in validation and value > validation['max']:
                    field_result["valid"] = False
                    field_result["messages"].append(f"Nilai maksimum: {validation['max']}")
                    validation_result["input_valid"] = False
                    if field not in validation_result["invalid_fields"]:
                        validation_result["invalid_fields"].append(field)
                
                validation_result["field_validation"][field] = field_result
        
        return jsonify(validation_result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/examples', methods=['GET'])
def get_examples():
    """Endpoint untuk mendapatkan contoh data input"""
    examples = {
        "mahasiswa_berprestasi": {
            "description": "Contoh mahasiswa dengan performa tinggi",
            "data": {
                'age': 20,
                'gender_encoded': 1,
                'attendance_percentage': 95,
                'mental_health_rating': 8.5,
                'exam_score': 88
            }
        },
        "mahasiswa_kesulitan": {
            "description": "Contoh mahasiswa yang mengalami kesulitan",
            "data": {
                'age': 19,
                'gender_encoded': 0,
                'attendance_percentage': 60,
                'mental_health_rating': 5,
                'exam_score': 55
            }
        },
        "mahasiswa_seimbang": {
            "description": "Contoh mahasiswa dengan kehidupan seimbang",
            "data": {
                'age': 21,
                'gender_encoded': 1,
                'attendance_percentage': 85,
                'mental_health_rating': 7,
                'exam_score': 75
            }
        },
        "input_minimal": {
            "description": "Data minimal yang diperlukan",
            "data": {
                'age': 20,
                'gender_encoded': 1,
                'attendance_percentage': 85,
                'mental_health_rating': 7,
                'exam_score': 75
            }
        }
    }
    
    return jsonify({
        "status": "success",
        "examples": examples,
        "usage": "Gunakan salah satu contoh data di atas untuk endpoint /predict",
        "note": "Hanya 5 field wajib: age, gender_encoded, attendance_percentage, mental_health_rating, exam_score"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint health check untuk monitoring"""
    try:
        model_status = "loaded" if model_loader is not None else "not_loaded"
        
        # Test prediksi sederhana
        test_successful = False
        if model_loader is not None:
            try:
                test_data = {
                    'age': 20,
                    'gender_encoded': 1,
                    'attendance_percentage': 80,
                    'mental_health_rating': 7,
                    'exam_score': 75
                }
                predictions, _ = model_loader.predict(test_data)
                test_successful = predictions is not None
            except:
                test_successful = False
        
        # Cek folder saved_models
        models_folder_exists = os.path.exists("saved_models")
        saved_models = []
        if models_folder_exists:
            try:
                saved_models = [f for f in os.listdir("saved_models") 
                              if os.path.isdir(os.path.join("saved_models", f))]
            except:
                saved_models = []
        
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "prediction_test": "passed" if test_successful else "failed",
            "models_folder_exists": models_folder_exists,
            "saved_models_count": len(saved_models),
            "latest_model": saved_models[0] if saved_models else None,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        })
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handler untuk 404 errors"""
    return jsonify({
        "error": "Endpoint tidak ditemukan",
        "status": "error",
        "available_endpoints": [
            "GET /",
            "GET /model/info", 
            "POST /predict",
            "POST /predict/batch",
            "POST /validate/input",
            "GET /examples",
            "GET /health"
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handler untuk 405 errors"""
    return jsonify({
        "error": "Method tidak diizinkan",
        "status": "error",
        "message": "Cek method HTTP yang digunakan (GET/POST)"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handler untuk 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "status": "error",
        "message": "Terjadi kesalahan pada server"
    }), 500

if __name__ == '__main__':
    print("Memulai Flask API...")
    print("=" * 50)
    
    # Inisialisasi model
    initialize_model()
    
    print("API siap digunakan!")
    print("\nEndpoint yang tersedia:")
    print("   GET  /              - Home page & status")
    print("   GET  /model/info    - Informasi model")
    print("   POST /predict       - Prediksi single mahasiswa")
    print("   POST /predict/batch - Prediksi batch mahasiswa")
    print("   POST /validate/input- Validasi data input")
    print("   GET  /examples      - Contoh data input")
    print("   GET  /health        - Health check")
    
    print(f"\nServer akan berjalan di: http://localhost:5000")
    print("Dokumentasi API: Akses GET / untuk melihat semua endpoint")
    print("Test API: Gunakan GET /examples untuk contoh data")
    
    print(f"\nTips penggunaan:")
    print("   • Minimal field: age, gender_encoded, attendance_percentage, mental_health_rating, exam_score")
    print("   • Field lain akan diisi otomatis jika kosong")
    print("   • Gunakan /validate/input untuk cek data sebelum prediksi")
    print("   • Gunakan /examples untuk melihat contoh data lengkap")
    
    # Jalankan Flask development server
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
    except Exception as e:
        print(f"Error menjalankan server: {str(e)}")
        print("Pastikan port 5000 tidak digunakan aplikasi lain")