import os


class Config:
    # File-handling
    UPLOAD_FOLDER = "uploads"
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB

    # Model artefact locations
    MODEL_PATH = "xgb_model.pkl"
    SCALER_PATH = "scaler.pkl"
    ENCODER_PATH = "encoder.pkl"

    # EXACT feature list (and order) used during training/scaler fit.
    # Keep this aligned with scaler.feature_names_in_.
    EXPECTED_FEATURES = [
        "DayOfWeek",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "CompetitionDistance",
        "CompetitionOpenNumMonths",
        "Promo2NumWeeks",
        "WeekOfYear",
        "PromoInterval_0",
        "PromoInterval_Feb,May,Aug,Nov",
        "PromoInterval_Mar,Jun,Sept,Dec",
        "StoreType_a",
        "StoreType_b",
        "StoreType_d",
        "Assortment_a",
        "Assortment_c",
    ]

    @classmethod
    def create_upload_folder(cls) -> None:
        if not os.path.exists(cls.UPLOAD_FOLDER):
            os.makedirs(cls.UPLOAD_FOLDER)
