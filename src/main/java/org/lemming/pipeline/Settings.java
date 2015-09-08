package org.lemming.pipeline;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;

/**
 * A class to store global parameters.
 */
public class Settings {
	
	// parameters
	public volatile Map<String,Integer> numberOfElements = new HashMap<>();
	
	static public Map<String, List<Double>> readCSV(String path){
		final Locale curLocale = Locale.getDefault();
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);
		
		List<String> list = new ArrayList<>();
		List<Double> param = new ArrayList<>(); 
		List<Double> zgrid = new ArrayList<>();
		List<Double> Calibcurve = new ArrayList<>();
		Map<String,List<Double>> result = new HashMap<>();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));

			String line;
			while ((line=br.readLine())!=null){
				if (line.contains("--")) break;
				list.add(line);
			}
			
			if ((line=br.readLine())!=null){
				String[] s = line.split(",");
				for (int i = 0; i < s.length; i++)
					param.add(Double.parseDouble(s[i].trim()));
			}
			br.close();
			
			for (String t : list){
				String[] splitted = t.split(",");
				zgrid.add(Double.parseDouble(splitted[0].trim()));
				Calibcurve.add(Double.parseDouble(splitted[3].trim()));
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		result.put("param", param);
		result.put("zgrid", zgrid);
		result.put("Calibcurve", Calibcurve);

		Locale.setDefault(curLocale);
		return result;
	}

	public static List<Double> readProps(String path) {
		final Locale curLocale = Locale.getDefault();
		final Locale usLocale = new Locale( "en", "US" ); // setting us locale
		Locale.setDefault( usLocale );
		
		List<Double> params = new ArrayList<>();
		try {
			FileReader reader = new FileReader( new File(path) );
			final Properties props = new Properties();
			props.load( reader );
			String[] paramParser = props.getProperty( "FitParameter", "" ).split( "[,\n]" );
			for (int i=0; i<paramParser.length; i++)
				params.add(Double.parseDouble(paramParser[i].trim()));
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Locale.setDefault( curLocale );
		return params;
	}
}
