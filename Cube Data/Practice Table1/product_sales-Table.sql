1. ROLLUP (Hierarchical Aggregation)

SELECT continent, country, city, SUM(units_sold) AS total_units_sold
FROM product_sales
GROUP BY ROLLUP(continent, country, city);

2. DICE (Subsetting a Cube)

SELECT continent, country, city, SUM(units_sold) AS total_units_sold
FROM product_sales
WHERE continent IN ('North America', 'Europe') AND country IN ('USA', 'Germany')
GROUP BY continent, country, city;

3. SLICE (Filtering a Slice of the Cube)

SELECT country, city, units_sold
FROM product_sales
WHERE continent = 'Asia';

4. DRILLDOWN (Detailed View of Data)

SELECT continent, country, city, units_sold
FROM product_sales
WHERE continent = 'Africa'
ORDER BY country, city, units_sold DESC;

5. PIVOT (Reshaping Data)

SELECT continent,
       SUM(CASE WHEN country = 'USA' THEN units_sold ELSE 0 END) AS USA_units_sold,
       SUM(CASE WHEN country = 'Germany' THEN units_sold ELSE 0 END) AS Germany_units_sold,
       SUM(CASE WHEN country = 'Japan' THEN units_sold ELSE 0 END) AS Japan_units_sold
FROM product_sales
GROUP BY continent;