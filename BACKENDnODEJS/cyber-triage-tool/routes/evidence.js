// routes/evidence.js
const express = require("express");
const router = express.Router();
const Evidence = require("../models/Evidence");

// CREATE: Add new evidence
router.post("/", async (req, res) => {
  const { title, description, investigator, caseId, type } = req.body;

  try {
    const newEvidence = new Evidence({ title, description, investigator, caseId, type });
    const savedEvidence = await newEvidence.save();
    res.status(201).json(savedEvidence);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// READ: Get all evidence
router.get("/", async (req, res) => {
  try {
    const evidenceList = await Evidence.find();
    res.status(200).json(evidenceList);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// READ: Get single evidence by ID
router.get("/:id", async (req, res) => {
  try {
    const evidence = await Evidence.findById(req.params.id);
    if (!evidence) return res.status(404).json({ message: "Evidence not found" });
    res.status(200).json(evidence);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// UPDATE: Update evidence
router.put("/:id", async (req, res) => {
  try {
    const updatedEvidence = await Evidence.findByIdAndUpdate(req.params.id, req.body, { new: true });
    if (!updatedEvidence) return res.status(404).json({ message: "Evidence not found" });
    res.status(200).json(updatedEvidence);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// DELETE: Delete evidence
router.delete("/:id", async (req, res) => {
  try {
    const deletedEvidence = await Evidence.findByIdAndDelete(req.params.id);
    if (!deletedEvidence) return res.status(404).json({ message: "Evidence not found" });
    res.status(200).json({ message: "Evidence deleted" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
