from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
# url = urlopen('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2019.zip')
with open("aiq.csv", 'w') as aiq:
    for year in range(1980, 2020):
        url = urlopen(f"https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_{year}.zip")
        with ZipFile(BytesIO(url.read())) as my_zip_file:
            for contained_file in my_zip_file.namelist():
                # with open(("unzipped_and_read_" + contained_file + ".file"), "wb") as output:
                for line in my_zip_file.open(contained_file).readlines():
                    print(line)
                    writableLine = line.decode("utf-8")
                    aiq.write(writableLine)
