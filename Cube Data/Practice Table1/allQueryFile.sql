1. ROLLUP --->

SELECT Country, Gender, SUM(Salary) AS Total_Salary
FROM Employees
GROUP BY ROLLUP(Country, Gender);

2. DICE (Subsetting a Cube) --->

SELECT Country, Gender, SUM(Salary) AS Total_Salary
FROM Employees
WHERE Country IN ('USA', 'India') AND Gender IN ('Male', 'Female')
GROUP BY Country, Gender;

3. SLICE (Filtering a Slice of the Cube) --->

SELECT Name, Gender, Salary
FROM Employees
WHERE Country = 'USA';

4. DRILLDOWN (Detailed View of Data) --->

SELECT Country, Gender, Name, Salary
FROM Employees
WHERE Country = 'USA'
ORDER BY Gender, Salary DESC;

5. PIVOT (Reshaping Data) --->

SELECT Country,
       SUM(CASE WHEN Gender = 'Male' THEN Salary ELSE 0 END) AS Male_Salary,
       SUM(CASE WHEN Gender = 'Female' THEN Salary ELSE 0 END) AS Female_Salary
FROM Employees
GROUP BY Country;

