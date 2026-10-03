# progetto_pilota_finance
finanza familiare
import pandas as pd

wrangler_sample_df = pd.read_csv("https://aka.ms/wrangler/titanic.csv")
display(wrangler_sample_df)

from pyspark.sql.functions import col, to_date, trim, abs, when

# 1. Caricamento tabelle Bronze (con lo schema dbo)
df_movimenti = spark.read.table("lh_personal_finance.dbo.bronze_movimenti_banca")
df_budget = spark.read.table("lh_personal_finance.dbo.bronze_budget_mensile")
df_categorie = spark.read.table("lh_personal_finance.dbo.bronze_anagrafica_categorie")

# 2. Trasformazione Movimenti
silver_movimenti = df_movimenti \
    .withColumn("TransactionID", trim(col("TransactionID"))) \
    .withColumn("Data", to_date(col("Data"))) \
    .withColumn("Conto", trim(col("Conto"))) \
    .withColumn("Categoria", trim(col("Categoria"))) \
    .withColumn("Sottocategoria", trim(col("Sottocategoria"))) \
    .withColumn("Importo", col("Importo").cast("decimal(18,2)")) \
    .withColumn("TipoMovimento", when(col("Importo") >= 0, "Entrata").otherwise("Uscita")) \
    .withColumn("Descrizione", trim(col("Descrizione")))

# 3. Trasformazione Budget
silver_budget = df_budget \
    .withColumn("AnnoMese", trim(col("AnnoMese"))) \
    .withColumn("Categoria", trim(col("Categoria"))) \
    .withColumn("Sottocategoria", trim(col("Sottocategoria"))) \
    .withColumn("TargetSpesa", abs(col("TargetSpesa").cast("decimal(18,2)")))

# 4. Trasformazione Categorie
silver_categorie = df_categorie \
    .withColumn("Categoria", trim(col("Categoria"))) \
    .withColumn("Sottocategoria", trim(col("Sottocategoria"))) \
    .withColumn("TipoCategoria", trim(col("TipoCategoria"))) \
    .withColumn("Note", trim(col("Note")))

# 5. Salvataggio Tabelle Silver nello schema dbo
silver_movimenti.write.mode("overwrite").format("delta").saveAsTable("lh_personal_finance.dbo.silver_movimenti_banca")
silver_budget.write.mode("overwrite").format("delta").saveAsTable("lh_personal_finance.dbo.silver_budget_mensile")
silver_categorie.write.mode("overwrite").format("delta").saveAsTable("lh_personal_finance.dbo.silver_anagrafica_categorie")

print("Trasformazione Silver completata con successo!")

from pyspark.sql.functions import (
    col, date_format, year, month, quarter, expr, 
    dense_rank, monotonically_increasing_id, lit, abs
)
from pyspark.sql.window import Window

# 1. Caricamento tabelle dal Silver Layer
df_movimenti = spark.read.table("lh_personal_finance.dbo.silver_movimenti_banca")
df_budget = spark.read.table("lh_personal_finance.dbo.silver_budget_mensile")
df_categorie = spark.read.table("lh_personal_finance.dbo.silver_anagrafica_categorie")

# ==========================================
# 2. CREAZIONE DIMENSIONE: gold_d_category
# ==========================================
gold_d_category = df_categorie \
    .select("Categoria", "Sottocategoria", "TipoCategoria", "Note") \
    .distinct() \
    .withColumn("CategoryKey", monotonically_increasing_id() + 1) \
    .select(
        col("CategoryKey").cast("int"),
        col("Categoria").alias("CategoryName"),
        col("Sottocategoria").alias("SubcategoryName"),
        col("TipoCategoria").alias("CategoryType"),
        col("Note")
    )

# ==========================================
# 3. CREAZIONE DIMENSIONE: gold_d_account
# ==========================================
gold_d_account = df_movimenti \
    .select("Conto") \
    .distinct() \
    .withColumn("AccountKey", monotonically_increasing_id() + 1) \
    .select(
        col("AccountKey").cast("int"),
        col("Conto").alias("AccountName")
    )

# ==========================================
# 4. CREAZIONE DIMENSIONE: gold_d_calendar
# ==========================================
# Generazione dinamica del calendario in base alle date presenti nei movimenti
min_max_dates = df_movimenti.selectExpr("min(Data) as min_date", "max(Data) as max_date").collect()[0]
min_date = min_max_dates["min_date"] or "2026-01-01"
max_date = min_max_dates["max_date"] or "2026-12-31"

gold_d_calendar = spark.sql(f"""
    SELECT 
        CAST(date_format(calendar_date, 'yyyyMMdd') AS INT) AS DateKey,
        calendar_date AS Date,
        YEAR(calendar_date) AS Year,
        MONTH(calendar_date) AS MonthNumber,
        date_format(calendar_date, 'MMMM') AS MonthName,
        date_format(calendar_date, 'yyyy-MM') AS YearMonth,
        QUARTER(calendar_date) AS Quarter
    FROM (
        SELECT explode(sequence(to_date('{min_date}'), to_date('{max_date}'), interval 1 day)) AS calendar_date
    )
""")

# ==========================================
# 5. CREAZIONE TABELLA FATTI: gold_f_expense_income
# ==========================================
gold_f_expense_income = df_movimenti \
    .join(gold_d_category, 
          (df_movimenti.Categoria == gold_d_category.CategoryName) & 
          (df_movimenti.Sottocategoria == gold_d_category.SubcategoryName), "left") \
    .join(gold_d_account, df_movimenti.Conto == gold_d_account.AccountName, "left") \
    .withColumn("DateKey", date_format(col("Data"), "yyyyMMdd").cast("int")) \
    .select(
        col("TransactionID"),
        col("DateKey"),
        col("AccountKey"),
        col("CategoryKey"),
        col("Importo").alias("Amount"),
        col("TipoMovimento").alias("TransactionType"),
        col("Descrizione").alias("Description")
    )

# ==========================================
# 6. CREAZIONE TABELLA FATTI: gold_f_budget
# ==========================================
gold_f_budget = df_budget \
    .join(gold_d_category, 
          (df_budget.Categoria == gold_d_category.CategoryName) & 
          (df_budget.Sottocategoria == gold_d_category.SubcategoryName), "left") \
    .select(
        col("AnnoMese").alias("YearMonth"),
        col("CategoryKey"),
        col("TargetSpesa").alias("TargetAmount")
    )

# ==========================================
# 7. SALVATAGGIO TABELLE GOLD (DELTA TABLES)
# ==========================================
gold_d_category.write.mode("overwrite").format("delta").saveAsTable("lh_personal_finance.dbo.gold_d_category")
gold_d_account.write.mode("overwrite").format("delta").saveAsTable("lh_personal_finance.dbo.gold_d_account")
gold_d_calendar.write.mode("overwrite").format("delta").saveAsTable("lh_personal_finance.dbo.gold_d_calendar")
gold_f_expense_income.write.mode("overwrite").format("delta").saveAsTable("lh_personal_finance.dbo.gold_f_expense_income")
gold_f_budget.write.mode("overwrite").format("delta").saveAsTable("lh_personal_finance.dbo.gold_f_budget")

print("Gold Layer generato con successo! Star Schema pronto.")
spark.sql("CREATE TABLE IF NOT EXISTS lh_personal_finance.dbo.gold_misure (ID INT)").write.mode("overwrite").format("delta").saveAsTable("lh_personal_finance.dbo.gold_misure")
