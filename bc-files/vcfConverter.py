# default

begin = "BEGIN:VCARD\n"
version = "VERSION:4.0\n"
end = "END:VCARD"

# param
lastname = "Patel"
firstname = "Rahul"
prefix = "Mr."

company = "DesDevs"
job_title = "Owner"
pic = "image/gif:http://www.example.com/dir_photos/my_photo.gif"

phone = "908-555-5555"
fax = "908-111-1111"

address = "55 Hoes Mad Lane;Baytown;LA;30314"
e_mail = "mail@desdevs.com"

# fields
n = "N:" + lastname + ";" + firstname + ";;" + prefix + ";\n"
fn = "FN:" + firstname + " " + lastname + "\n"
org = "ORG:" + company + "\n"
title = "TITLE:" + job_title + "\n"
photo = "PHOTO;MEDIATYPE=" + pic + "\n"
tel = "TEL:" + phone + "\n"
tel_fax = "TEL;TYPE=FAX:" + fax + "\n"
adr = "ADR:;" + address + "\n"
email = "EMAIL:" + e_mail + "\n"

vcard = (begin + version + n + fn + org + title + photo + tel + tel_fax + adr + email + end)


with open('C://Users//HHS//Desktop//out.vcf', 'w') as f:
    print(vcard, file=f)