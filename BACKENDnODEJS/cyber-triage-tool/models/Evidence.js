// models/Evidence.js
const mongoose = require("mongoose");

const EvidenceSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true,
  },
  description: {
    type: String,
    required: true,
  },
  investigator: {
    type: String,
    required: true,
  },
  caseId: {
    type: String,
    required: true,
  },
  collectedAt: {
    type: Date,
    default: Date.now,
  },
  type: {
    type: String,
    required: true, // File, Registry, Log, Network Activity, etc.
  },
  status: {
    type: String,
    default: "pending", // pending, analyzed, suspicious, etc.
  },
  IOCs: {
    type: [String], // List of Indicators of Compromise
  },
});

module.exports = mongoose.model("Evidence", EvidenceSchema);
