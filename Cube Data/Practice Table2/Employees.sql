CREATE TABLE Employees (
  Id INT PRIMARY KEY,
  Name VARCHAR(50),
  Gender VARCHAR(10),
  Salary INT,
  Country VARCHAR(50)
);

INSERT INTO Employees VALUES (1, 'Mark', 'Male', 5000, 'USA');
INSERT INTO Employees VALUES (2, 'John', 'Male', 4500, 'India');
INSERT INTO Employees VALUES (3, 'Pam', 'Female', 5500, 'USA');
INSERT INTO Employees VALUES (4, 'Sara', 'Female', 4000, 'India');
INSERT INTO Employees VALUES (5, 'Todd', 'Male', 3500, 'India');
INSERT INTO Employees VALUES (6, 'Mary', 'Female', 5000, 'UK');
INSERT INTO Employees VALUES (7, 'Ben', 'Male', 6500, 'UK');
INSERT INTO Employees VALUES (8, 'Elizabeth', 'Female', 7000, 'USA');
INSERT INTO Employees VALUES (9, 'Tom', 'Male', 5500, 'UK');
INSERT INTO Employees VALUES (10, 'Ron', 'Male', 5000, 'USA');

-------------------------------------------------------------

Main Query :
------------
Select Name, NULL, NULL, Sum(Salary) AS TotalSalary
From Employees
GROUP BY Name

UNION ALL

Select NULL, Gender, NULL, Sum(Salary) AS TotalSalary
From Employees
GROUP BY Gender

UNION ALL

Select NULL, NULL, Country, Sum(Salary) AS TotalSalary 
From Employees
GROUP By Country

UNION ALL

Select Name, Gender, NULL, Sum(Salary) AS TotalSalary
From Employees
GROUP BY Name, Gender

UNION ALL

Select Name, NULL, Country, Sum(Salary) AS TotalSalary
From Employees
GROUP By Name, Country

UNION ALL

Select NULL, Gender, Country, Sum(Salary) AS TotalSalary
From Employees
GROUP BY Gender, Country

UNION ALL

Select Name, Gender, Country, Sum(Salary) AS TotalSalary
From Employees
GROUP By Name, Gender, Country

UNION ALL

Select NULL, NULL, NULL, Sum(Salary) AS TotalSalary
From Employees;

-----------------------------------------------------------

AnotherWay :
------------
Select Name, NULL AS Gender, NULL AS Country, Sum(Salary) AS TotalSalary
From Employees
GROUP BY Name

UNION ALL

Select NULL AS Name, Gender, NULL AS Country, Sum(Salary) AS TotalSalary
From Employees
GROUP BY Gender

UNION ALL

Select NULL AS Name, NULL AS Gender, Country, Sum(Salary) AS TotalSalary 
From Employees
GROUP By Country

UNION ALL

Select Name, Gender, NULL AS Country, Sum(Salary) AS TotalSalary
From Employees
GROUP BY Name, Gender

UNION ALL

Select Name, NULL AS Gender, Country, Sum(Salary) AS TotalSalary
From Employees
GROUP By Name, Country

UNION ALL

Select NULL AS Name, Gender, Country, Sum(Salary) AS TotalSalary
From Employees
GROUP BY Gender, Country

UNION ALL

Select Name, Gender, Country, Sum(Salary) AS TotalSalary
From Employees
GROUP By Name, Gender, Country

UNION ALL

Select NULL AS Name, NULL AS Gender, NULL AS Country, Sum(Salary) AS TotalSalary
From Employees;

------------------------------------------------------------

GROUPING SETS:
--------------
Select Name, Gender, Country, Sum(Salary) AS TotalSalary
From Employees
GROUP BY
GROUPING SETS (
  (Name),
  (Gender),
  (Country),
  (Name, Gender), 
  (Name, Country), 
  (Gender, Country), 
  (Name, Gender, Country),
  ()
);

------------------------------------------------------------

AnotherWay GROUPING SETS:
-------------------------
Select Name, Gender, Country, Sum(Salary) AS TotalSalary
From Employees
GROUP BY
GROUPING SETS (
  (Name),
  (Gender),
  (Country),
  (Name, Gender), 
  (Name, Country), 
  (Gender, Country), 
  (Name, Gender, Country),
  ()
)
ORDER BY GROUPING(Country), GROUPING(Gender), Gender;

------------------------------------------------------------

ROLL_UP Query:
--------------
Select Country, Sum(Salary) AS TotalSalary
From Employees
GROUP BY ROLLUP(Country);

------------------------------------------------------------

AnotherWay ROLL_UP:
-------------------
SELECT NULL AS Name, NULL AS Gender, Country, Sum(Salary)
FROM Employees
GROUP BY Country

UNION ALL

SELECT NULL AS Name, NULL AS Gender, NULL AS Country, Sum(Salary)
FROM Employees;


AnotherWay ROLL_UP:
-------------------
Select Country, Sum(Salary) AS TotalSalary
From Employees
GROUP BY Country with ROLLUP;


AnotherWay ROLL_UP:
-------------------
SELECT 
    Country,
    Gender,
    SUM(Salary) AS Total_Salary
FROM 
    Employees
GROUP BY 
    ROLLUP (Country, Gender)
ORDER BY 
    Country,
    Gender;


------------------------------------------------------------

1.
Select Country, Gender, Sum(Salary) as TotalSalary
From Employees
Group By Country, Gender

------------------------------------------------------------

2.
Select Country, Gender, Sum(Salary) as TotalSalary
From Employees
Group By Country, Gender

UNION ALL 

Select Country, NULL, Sum(Salary) as TotalSalary
From Employees 
Group By Country

------------------------------------------------------------

3.
SELECT Name, Gender, Country, SUM(Salary) AS total_salary
FROM Employees
GROUP BY CUBE(Name, Gender, Country);

------------------------------------------------------------

4.
SELECT NULL, Gender, Sum(Salary) AS total_salary
From Employees
GROUP BY Gender;

------------------------------------------------------------

5.
SELECT Name, Gender, Country, sum(Salary) AS TotalSalary
FROM Employees
GROUP BY 
CUBE(Name, Gender, Country)
HAVING 
sum(Salary) > 10000;

------------------------------------------------------------

6. Pivot Query:
SELECT 
    Country,
    SUM(CASE WHEN Gender = 'Male' THEN Salary ELSE 0 END) AS Male_Total_Salary,
    SUM(CASE WHEN Gender = 'Female' THEN Salary ELSE 0 END) AS Female_Total_Salary
FROM 
    Employees
GROUP BY 
    Country
ORDER BY 
    Country;
