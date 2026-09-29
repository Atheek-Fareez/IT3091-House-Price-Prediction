# IT3091 MACHINE LEARNING – INITIAL GROUP PROJECT PLAN

## House Price Prediction and Valuation Feature Analysis

**Group Code:** 8 | **Sector:** Real Estate | **Dataset:** Ames Housing / House Prices

### Project Item Choice

| Item | Choice |
|---|---|
| Track | Guided Data Track |
| Primary Lens | Price Prediction |
| Secondary Lens | Valuation Feature Analysis |
| Exact ML Task | Supervised regression |
| Unit of Analysis | One property / house record |
| Main Output | Predicted house sale price |
| Stakeholder | Real estate company / property valuation decision-makers |

---

## 1. Project Idea

The project will use the Ames Housing / House Prices dataset to develop and compare machine-learning regression methods for estimating house sale prices from property characteristics. Price Prediction is the primary decision lens. Valuation Feature Analysis is used as a secondary lens to explain which property characteristics are most associated with value and to strengthen the pricing recommendation.

---

## 2. Business Problem

A real estate company wants to understand property value factors and support better pricing decisions. The project therefore asks whether historical housing data can be used to estimate sale prices accurately and explain the major factors associated with valuation.

---

## 3. Proposed Task and Output

| Element | Proposed Description |
|---|---|
| Task | Train and compare supervised regression models for house-price prediction. |
| Target / Output | Predicted house sale price. |
| Secondary Analysis | Analyse influential valuation features after modelling. |
| Recommendation | Recommend the most defensible model/method and explain how the findings support pricing decisions. |

---

## 4. End-to-End Workflow

```
Business Problem & Lens
    ↓
Data Understanding + EDA
    ↓
Data Quality Reasoning
    ↓
Preprocessing + Feature Engineering
    ↓
Train/Test or Cross-Validation Logic
    ↓
Baseline + At Least Three Alternative Models
    ↓
Evaluation & Comparison
    ↓
Valuation Feature Analysis
    ↓
Recommendation + Limitations + Documentation
```

---

## 5. Balanced Division Among Four Members

The assignment descriptor does not prescribe a fixed four-member split. This division is designed to map the four members to the major evidence areas in the marking rubric while keeping the work connected. The workflow diagram and decision log are shared group evidence.

| Member | Core Responsibility | Main Deliverables / Evidence |
|---|---|---|
| Member 1 | Business Problem Framing, Lens & Task Formulation | Stakeholder; decision need; primary and secondary lens with reasons; unit of analysis; exact regression task/output; rationale; contribution to problem-framing canvas and early decision log. |
| Member 2 | Data Understanding, EDA & Data Quality Reasoning | Data dictionary; row meaning; variable types; target exploration; distributions/relationships; missing values; duplicates/outliers/data issues; EDA insight log; explain how issues affect the task. |
| Member 3 | Preprocessing & Feature Engineering | Justified missing-value handling; categorical encoding; scaling where needed; outlier decisions; leakage checks; feature engineering/selection; reproducible preprocessing pipeline; preprocessing/feature log. |
| Member 4 | Model Strategy, Evaluation & Recommendation | Sensible baseline + at least three alternatives; train/test or cross-validation logic; regression metrics; model comparison; valuation-feature interpretation; final recommendation, limitations and stakeholder value. |

---

## 6. Member 1 – Business Problem Framing, Lens & Task Formulation

This member owns the foundation of the entire ML project. The work should include:

- Clearly identify the stakeholder: a real estate company / property valuation decision-makers.
- State the decision need: improve and support property pricing decisions using historical housing data.
- Select Price Prediction as the primary lens and justify why it directly addresses the decision need.
- Select Valuation Feature Analysis as the optional secondary lens and explain how it strengthens price prediction.
- Define the unit of analysis as one property / house record.
- Define the exact ML task as supervised regression.
- Define the expected output as a predicted house sale price.
- Explain the business rationale and expected stakeholder value.
- Contribute these decisions to the problem-framing canvas and decision log.

---

## 7. Member 2 – Data Understanding, EDA & Data Quality

- Understand what each dataset row represents and identify the target and candidate predictors.
- Prepare the data dictionary and classify variable types.
- Explore the sale-price target and important property-variable distributions.
- Study meaningful relationships between variables and the target.
- Identify missing values, duplicates, unusual values and possible outliers.
- Record important findings in the EDA insight log.
- Explain how discovered data issues may affect modelling decisions.

---

## 8. Member 3 – Preprocessing & Feature Engineering

- Choose and justify missing-value handling.
- Choose suitable encoding for categorical variables.
- Decide when scaling or transformations are required.
- Investigate and justify treatment of outliers.
- Check for data leakage and ensure preprocessing is fitted correctly.
- Develop/select useful features based on evidence rather than automatically.
- Maintain the preprocessing/feature log and reproducible preprocessing pipeline.

---

## 9. Member 4 – Model Strategy, Evaluation & Recommendation

- Build a sensible regression baseline.
- Compare at least three alternative models/methods.
- Use suitable train/test or cross-validation logic.
- Select and justify appropriate regression evaluation metrics.
- Compare models honestly rather than relying on one score.
- Interpret valuation features using suitable model/analysis evidence.
- Prepare the final evidence-based recommendation.
- State limitations, risks and stakeholder value.

---

## 10. Shared Group Responsibilities

Some evidence connects all four areas and should be reviewed together rather than assigned to only one person.

- **Workflow diagram** – shows the coherent path from the business problem to the recommendation.
- **Decision log** – each member records important choices, alternatives, reasons and evidence from their own section.
- **Model/method comparison** – Member 4 leads it, but preprocessing and EDA decisions from Members 2 and 3 must be reflected.
- **Final recommendation and limitations** – Member 4 leads, but the recommendation must be supported by the whole group's evidence.
- **Reproducibility/documentation** – all members keep their work clear so notebook outputs and report statements agree.
- **AI-use transparency** – the group honestly declares how AI tools were used.

---

## 11. What Is Required for the Initial Submission

For the initial group submission, include:

- Track: Guided Data Track.
- Primary lens and optional secondary lens, with reasons.
- Proposed exact task and output.
- Proposed workflow.
- Core responsibility of each of the four members.

The full EDA, preprocessing, trained models and final evaluation are not required as completed results in the initial proposal; they are the planned work that will be developed during the project.

---

## 12. Alignment With the Marking Areas

| Rubric Area | Main Owner | How the Team Covers It |
|---|---|---|
| Business problem framing and lens/task formulation | Member 1 | Stakeholder, decision need, lenses, unit of analysis, exact task/output and rationale. |
| Workflow diagram and decision log | Shared | One coherent workflow; each member contributes evidence-based decisions. |
| Data understanding, EDA and data quality reasoning | Member 2 | Data dictionary, variable types, patterns and data issues. |
| Preprocessing and feature-engineering decisions | Member 3 | Missing values, outliers, categories, scaling, leakage and feature decisions. |
| Model/data-mining strategy and comparison | Member 4 | Baseline plus at least three alternatives and justified comparison. |
| Evaluation, validation and critical judgement | Member 4 | Appropriate metrics, validation logic, leakage awareness and honest interpretation. |
| Recommendation, limitations and stakeholder value | Member 4 + Shared review | Evidence-based recommendation, limitations and practical value. |
| Reproducibility, documentation and AI-use transparency | Shared | Clear notebook/report consistency, dataset reference and honest AI declaration. |

---

## 13. Initial Proposal Summary

The group proposes a supervised regression project using the Ames Housing / House Prices dataset. The primary lens is Price Prediction and the secondary lens is Valuation Feature Analysis. The project will progress from business framing through data understanding, EDA, preprocessing, feature engineering, model comparison, validation, valuation interpretation and an evidence-based real-estate recommendation. Responsibilities are divided across four members according to the major rubric areas, with workflow, decision logging, documentation and final review treated as shared group evidence.
