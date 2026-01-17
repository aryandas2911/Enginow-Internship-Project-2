Predict whether a loan applicant will default (high risk) or not default (low risk), using demographic, financial, and credit history features — to reduce lender risk.

Target Variable: loan_status (0: low risk and 1: high risk) [Binary classification]
Focus on Recall & Precision, not just accuracy

person_age Age (Numerical) 
person_income Annual Income (Numerical) 
person_home_ownership Home ownership (Categorical)
person_emp_length Employment length (in years) (Numerical) 
loan_intent Loan intent (Categorical)
loan_grade Loan grade (Categorical)
loan_amnt Loan amount (Numerical) 
loan_int_rate Interest rate (Numerical) 
loan_status Loan status (0 is non default 1 is default) (Numerical) 
loan_percent_income Percent income (Numerical) 
cb_person_default_on_file Historical default (Categorical)
cb_preson_cred_hist_length Credit history length (Numerical)

EDA Analysis:
1. Most applicants have low risk
2. Applicants within age group 18-25 have the highest number with most defaults (high risk)
3. Applicants with low risk have higher annual income than applicants with high risk
4. Most applicants have loan grade as A followed by B, but, most high risk applicants are in loan grade D
5. There are more risky applicants with no historical default because that group is much larger overall. Proportionally, applicants with past defaults are still riskier.
6. 'loan_status' has low pearson correlation with all features
7. There are some outliers in applicants annual incomes
8. Most of the applicants have a loan requirement within 12500 to 5000 but there are some with high loan requirements like 35000 as well
9. Most applicants have got a loan within interest rate of 12.5 to 7.5 but some have recevied upto 22.5 as well