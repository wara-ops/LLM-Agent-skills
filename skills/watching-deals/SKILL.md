---
name: watching-deals
description: Get information on electronic equipment available at discount through daily deals
---

## Overview

This skill retrieves information about discounted electronic equipment.

## Instructions

Always use curl to retrive information. The base URL is "https://api.restful-api.dev/"

Note that even if neither price nor actual discount figures are given, all of the listed items are at some sort of discount.

Be aware that the user might ask for 'iPhone' but the retrieved data may be more specific e.g. 'Apple iPhone 12 Mini'. Make sure to list all such entries in that case.  

### Examples

1. List of all deals

```bash
curl -s "https://api.restful-api.dev/objects"
```

2. Detailed information on a particular deal

```bash
curl -s "https://api.restful-api.dev/objects/{id}"
```
The result is a JSON dictionary with all information for item {id}
