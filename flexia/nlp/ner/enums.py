from ...enums import EqualEnum, ExplicitEnum


class TaggingScheme(ExplicitEnum, EqualEnum):
    BIO = "bio"
    BIEO = "bieo"