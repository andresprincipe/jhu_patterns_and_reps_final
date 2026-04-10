import pathlib
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


TEAM_USA_BASE_URL = "https://www.teamusa.com/profiles/"
OLYMPIC_ATHLETES_CSV_PATH = (pathlib.Path(__file__).parent
                             / "raw_data"
                             / "olympic_medals.csv")

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def build_slug_from_name(full_name: str) -> str | None:
    """Build a first-last slug, ignoring middle names/initials and suffixes."""
    if pd.isna(full_name):
        return None

    cleaned_name = str(full_name).lower().replace("'", "")
    name_parts = re.findall(r"[a-z]+(?:-[a-z]+)?", cleaned_name)
    if len(name_parts) < 2:
        return None

    while len(name_parts) > 2 and name_parts[-1] in NAME_SUFFIXES:
        name_parts.pop()

    if len(name_parts) < 2:
        return None

    return f"{name_parts[0]}-{name_parts[-1]}"


def build_full_name_slug(full_name: str) -> str | None:
    """Build a slug using all name tokens (fallback when first-last slug misses)."""
    if pd.isna(full_name):
        return None

    cleaned_name = str(full_name).lower().replace("'", "")
    name_parts = re.findall(r"[a-z]+(?:-[a-z]+)?", cleaned_name)
    if not name_parts:
        return None

    return "-".join(name_parts)


def get_usa_athlete_names():

    medal_df = pd.read_csv(OLYMPIC_ATHLETES_CSV_PATH)
    cols_to_keep = ["athlete_full_name", "country_code"]
    name_df = medal_df[cols_to_keep].copy()
    name_df = name_df[name_df["country_code"] == "US"]
    name_df = name_df.drop_duplicates(subset=["athlete_full_name"])

    name_df["athlete_slug"] = name_df["athlete_full_name"].apply(build_slug_from_name)
    name_df.dropna(subset=["athlete_slug"], inplace=True)
    athlete_li = name_df["athlete_slug"].drop_duplicates().to_list()
    return athlete_li


def get_usa_athlete_slug_candidates() -> pd.DataFrame:
    """Return slug candidates per athlete to improve Team USA URL hit rate."""
    medal_df = pd.read_csv(OLYMPIC_ATHLETES_CSV_PATH)
    cols_to_keep = ["athlete_full_name", "country_code"]
    name_df = medal_df[cols_to_keep].copy()
    name_df = name_df[name_df["country_code"] == "US"]
    name_df = name_df.drop_duplicates(subset=["athlete_full_name"])

    name_df["athlete_slug_primary"] = name_df["athlete_full_name"].apply(build_slug_from_name)
    name_df["athlete_slug_fallback"] = name_df["athlete_full_name"].apply(build_full_name_slug)
    name_df.dropna(subset=["athlete_slug_primary"], inplace=True)
    return name_df[["athlete_full_name", "athlete_slug_primary", "athlete_slug_fallback"]].copy()


def build_team_usa_profile_url(athlete_slug: str) -> str:
    return f"{TEAM_USA_BASE_URL}{athlete_slug}"


def fetch_profile_soup(profile_url: str) -> BeautifulSoup:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(profile_url, headers=headers, timeout=20)
    response.raise_for_status()
    if not response.text.strip():
        raise ValueError(f"Empty response for profile URL: {profile_url}")
    return BeautifulSoup(response.text, "html.parser")


def scrape_field_from_soup(soup: BeautifulSoup, field_name: str) -> str | None:
    """Find an athlete profile field by heading text and return its value."""
    field_header = soup.find("h3", string=lambda t: t and field_name in t)
    if not field_header:
        return None

    field_paragraph = field_header.find_next("p")
    if not field_paragraph:
        return None

    return field_paragraph.get_text(strip=True)


def scrape_hometown(profile_url: str) -> str | None:
    """Scrape the hometown value from a Team USA athlete profile page."""
    soup = fetch_profile_soup(profile_url)
    return scrape_field_from_soup(soup, "Hometown")


def scrape_height(profile_url: str) -> str | None:
    """Scrape the height value from a Team USA athlete profile page."""
    soup = fetch_profile_soup(profile_url)
    return scrape_field_from_soup(soup, "Height")


def scrape_age(profile_url: str) -> str | None:
    """Scrape the age value from a Team USA athlete profile page."""
    soup = fetch_profile_soup(profile_url)
    return scrape_field_from_soup(soup, "Age")


def scrape_education(profile_url: str) -> str | None:
    """Scrape the education value from a Team USA athlete profile page."""
    soup = fetch_profile_soup(profile_url)
    return scrape_field_from_soup(soup, "Education")


def scrape_profile_details(profile_url: str) -> dict[str, str | None]:
    """Scrape common athlete profile fields from one Team USA page request."""
    soup = fetch_profile_soup(profile_url)
    return {
        "hometown": scrape_field_from_soup(soup, "Hometown"),
        "height": scrape_field_from_soup(soup, "Height"),
        "age": scrape_field_from_soup(soup, "Age"),
        "education": scrape_field_from_soup(soup, "Education"),
    }


def get_usa_athlete_profiles(
    limit: int | None = None,
    stop_on_missing_profile: bool = False) -> pd.DataFrame:
    slug_df = get_usa_athlete_slug_candidates()
    usa_athletes = slug_df.to_dict("records")
    if limit is not None:
        usa_athletes = usa_athletes[:limit]

    output_columns = [
        "athlete_full_name",
        "athlete_slug",
        "profile_url",
        "hometown",
        "height",
        "age",
        "education",
    ]
    rows = []
    for athlete_rec in tqdm(usa_athletes, desc="Scraping Profile Info"):
        athlete_name = athlete_rec["athlete_full_name"]
        slug_candidates = [
            athlete_rec.get("athlete_slug_primary"),
            athlete_rec.get("athlete_slug_fallback"),
        ]
        slug_candidates = [slug for slug in slug_candidates if slug]
        slug_candidates = list(dict.fromkeys(slug_candidates))

        selected_slug = slug_candidates[0] if slug_candidates else None
        selected_profile_url = build_team_usa_profile_url(selected_slug) if selected_slug else None
        profile_details = None

        for slug in slug_candidates:
            candidate_url = build_team_usa_profile_url(slug)
            try:
                profile_details = scrape_profile_details(candidate_url)
                selected_slug = slug
                selected_profile_url = candidate_url
                break
            except (requests.RequestException, ValueError):
                continue

        if profile_details is None:
            profile_details = {
                "hometown": None,
                "height": None,
                "age": None,
                "education": None,
            }
            if stop_on_missing_profile:
                tqdm.write(f"Stopping: unable to fetch profile for {athlete_name}")
                break

        rows.append(
            {
                "athlete_full_name": athlete_name,
                "athlete_slug": selected_slug,
                "profile_url": selected_profile_url,
                "hometown": profile_details["hometown"],
                "height": profile_details["height"],
                "age": profile_details["age"],
                "education": profile_details["education"],
            }
        )

    return pd.DataFrame(rows, columns=output_columns)

if __name__ == "__main__":
    profile_df = get_usa_athlete_profiles()
    profile_df.fillna("Unknown",inplace=True)
    OUTPUT_PATH = pathlib.Path(__file__).parent / "raw_data" / "athlete_profiles.csv"
    profile_df.to_csv(OUTPUT_PATH, index=False)
    print(profile_df.head(10))
    