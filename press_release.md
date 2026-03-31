# ML Triage Revolutionizes FDA Drug Safety Monitoring

## Catching Fatal Drug Interactions Before They Escalate
> Every day, the FDA receives thousands of adverse drug event reports. Buried within this mountain of data are critical warnings about life-threatening drug interactions, but human reviewers simply cannot read them fast enough to prevent ongoing harm. 

## The Bottleneck in Post-Market Drug Safety
The FDA's Adverse Event Reporting System (FAERS) is fundamentally overwhelmed by volume. Regulatory analysts face millions of voluntary reports ranging from mild nausea to fatal heart failure. Currently, sifting through this data is a reactive, manual process. The specific problem is alert fatigue: when an overloaded system treats a mild rash and a severe hemorrhage with the same initial urgency, critical safety signals are delayed. Regulators need a way to better automatically predict which incoming reports represent true, life-threatening emergencies based on complex patient profiles.\

## A Machine Learning Solution for Public Health
We have developed a data-driven Machine Learning solution that instantly evaluates incoming adverse event reports. Instead of waiting for a human to connect the dots, our ML triage system analyzes a patient's age, the number of concurrent medications they are taking (polypharmacy), and their exposure to known high-risk drugs. It then predicts severity, automatically flagging the highest-risk cases for immediate regulatory review. This ensures that the most dangerous drug complications are investigated first, prioritizing public health resources and streamlining pharmacovigilance (monitoring for adverse effects in post-market drugs).

## Visualizing the Solution: Model Performance
The following chart demonstrates the system's effectiveness. By evaluating precision (trustworthiness of the alerts) and recall (ability to catch all true emergencies), the Balanced Random Forest significantly outperforms baseline random chance. Specifically, the model achieves an overall performance score (AUC) of 0.574, representing around a 70% relative lift over the baseline severe-event incidence rate of 0.336. This proves the ML triage system can successfully isolate severe outcomes, and with refinements in feature selection (choosing how to inform the model) to further improve performance, we can ensure regulatory leads spend their time reviewing highly probable emergencies rather than false alarms.

[PR_Curve](./images/pr_curve.png)