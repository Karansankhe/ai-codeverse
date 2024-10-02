import os
import requests
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(page_title="Digital Forensics Evidence Manager", page_icon="🔍")

# API endpoint
API_URL = "http://localhost:5000/api/evidence"  # Update if your API URL is different

def fetch_evidence():
    response = requests.get(API_URL)
    return response.json() if response.status_code == 200 else []

def create_evidence(title, description, investigator, case_id, evidence_type):
    evidence_data = {
        "title": title,
        "description": description,
        "investigator": investigator,
        "caseId": case_id,
        "type": evidence_type
    }
    response = requests.post(API_URL, json=evidence_data)
    return response.json() if response.status_code == 201 else None

def update_evidence(evidence_id, updates):
    response = requests.put(f"{API_URL}/{evidence_id}", json=updates)
    return response.json() if response.status_code == 200 else None

def delete_evidence(evidence_id):
    response = requests.delete(f"{API_URL}/{evidence_id}")
    return response.status_code == 200

def main():
    st.title("Digital Forensics Evidence Manager")

    # Evidence creation section
    st.header("Add New Evidence")
    title = st.text_input("Title")
    description = st.text_area("Description")
    investigator = st.text_input("Investigator")
    case_id = st.text_input("Case ID")
    evidence_type = st.selectbox("Type", ["File", "Registry", "Log", "Network Activity"])

    if st.button("Add Evidence"):
        if title and description and investigator and case_id and evidence_type:
            result = create_evidence(title, description, investigator, case_id, evidence_type)
            if result:
                st.success("Evidence added successfully!")
            else:
                st.error("Failed to add evidence.")
        else:
            st.warning("Please fill in all fields.")

    # Display existing evidence
    st.header("Existing Evidence")
    evidence_list = fetch_evidence()

    if evidence_list:
        for evidence in evidence_list:
            st.subheader(evidence["title"])
            st.write(f"Description: {evidence['description']}")
            st.write(f"Investigator: {evidence['investigator']}")
            st.write(f"Case ID: {evidence['caseId']}")
            st.write(f"Type: {evidence['type']}")
            st.write(f"Collected At: {evidence['collectedAt']}")
            st.write(f"Status: {evidence['status']}")
            st.write(f"IOCs: {', '.join(evidence['IOCs']) if evidence.get('IOCs') else 'None'}")

            # Update and delete options
            if st.button(f"Update {evidence['title']}"):
                new_status = st.selectbox("Update Status", ["pending", "analyzed", "suspicious"])
                if st.button("Confirm Update"):
                    update_evidence(evidence["_id"], {"status": new_status})
                    st.success("Evidence updated successfully!")

            if st.button(f"Delete {evidence['title']}"):
                if delete_evidence(evidence["_id"]):
                    st.success("Evidence deleted successfully!")
                else:
                    st.error("Failed to delete evidence.")

    else:
        st.info("No evidence found.")

if __name__ == "__main__":
    main()
