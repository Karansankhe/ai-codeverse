from flask import Flask, jsonify, request, abort
from flask_pymongo import PyMongo
from bson import ObjectId
from datetime import datetime

app = Flask(__name__)

# MongoDB connection settings
app.config["MONGO_URI"] = "mongodb+srv://sankhe00009:rA1y8D1McvFVNeRY@cluster0.8ks4s.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongo = PyMongo(app)

# Utility function to format response data with proper ObjectId handling
def format_incident(incident):
    incident['_id'] = str(incident['_id'])  # Convert ObjectId to string
    return incident

# 1. Create an incident (POST)
@app.route('/incidents', methods=['POST'])
def create_incident():
    if not request.json or 'description' not in request.json:
        return jsonify({'error': 'Description is required'}), 400  # Bad request

    incident = {
        'description': request.json['description'],
        'status': request.json.get('status', 'New'),
        'created_at': datetime.utcnow().isoformat()  # Use ISO datetime format
    }
    
    # Insert the incident into the MongoDB collection
    result = mongo.db.incidents.insert_one(incident)
    incident['_id'] = str(result.inserted_id)  # Get the inserted ID
    return jsonify({'message': 'Incident created successfully', 'incident': format_incident(incident)}), 201

# 2. Get all incidents (GET)
@app.route('/incidents', methods=['GET'])
def get_incidents():
    incidents = list(mongo.db.incidents.find())
    incidents = [format_incident(incident) for incident in incidents]  # Format each incident
    return jsonify({'incidents': incidents})

# 3. Get a specific incident by ID (GET)
@app.route('/incidents/<string:incident_id>', methods=['GET'])
def get_incident(incident_id):
    try:
        incident = mongo.db.incidents.find_one({'_id': ObjectId(incident_id)})
        if incident is None:
            return jsonify({'error': 'Incident not found'}), 404
        return jsonify({'incident': format_incident(incident)})
    except:
        return jsonify({'error': 'Invalid incident ID format'}), 400  # Handle invalid ObjectId format

# 4. Update an incident (PUT)
@app.route('/incidents/<string:incident_id>', methods=['PUT'])
def update_incident(incident_id):
    try:
        incident = mongo.db.incidents.find_one({'_id': ObjectId(incident_id)})
        if incident is None:
            return jsonify({'error': 'Incident not found'}), 404

        if not request.json:
            return jsonify({'error': 'Request body must be JSON'}), 400

        # Update the incident data
        updated_data = {
            'description': request.json.get('description', incident['description']),
            'status': request.json.get('status', incident['status'])
        }
        mongo.db.incidents.update_one({'_id': ObjectId(incident_id)}, {'$set': updated_data})

        # Fetch updated incident
        updated_incident = mongo.db.incidents.find_one({'_id': ObjectId(incident_id)})
        return jsonify({'message': 'Incident updated successfully', 'incident': format_incident(updated_incident)})
    except:
        return jsonify({'error': 'Invalid incident ID format'}), 400

# 5. Delete an incident (DELETE)
@app.route('/incidents/<string:incident_id>', methods=['DELETE'])
def delete_incident(incident_id):
    try:
        result = mongo.db.incidents.delete_one({'_id': ObjectId(incident_id)})
        if result.deleted_count == 0:
            return jsonify({'error': 'Incident not found'}), 404
        return jsonify({'message': 'Incident deleted successfully'})
    except:
        return jsonify({'error': 'Invalid incident ID format'}), 400

# 6. Test data route (GET) - for quick testing
@app.route('/test', methods=['GET'])
def test_data():
    return jsonify({
        "create_incident": {
            "description": "Example incident description",
            "status": "New"
        },
        "update_incident": {
            "description": "Updated incident description",
            "status": "In Progress"
        },
        "example_incidents": [
            {
                "id": 1,
                "description": "Sample incident 1",
                "status": "New",
                "created_at": "2024-10-01T12:00:00"
            },
            {
                "id": 2,
                "description": "Sample incident 2",
                "status": "In Progress",
                "created_at": "2024-10-01T12:30:00"
            }
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
