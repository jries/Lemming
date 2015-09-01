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
	
	static public double[] readCSV(String path){
		final Locale curLocale = Locale.getDefault();
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);
		
		List<String> list = new ArrayList<>();
		double[] param = null; 
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));

			String line;
			String[] s;
			while ((line=br.readLine())!=null){
				if (line.contains("--")) break;
				list.add(line);
			}
			
			
			if ((line=br.readLine())!=null){
				s = line.split(",");
				param = new double[s.length];
				for (int i = 0; i < s.length; i++)
					param[i] = Double.parseDouble(s[i].trim());
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Locale.setDefault(curLocale);
		return param;
	}

	public static double[] readProps(String path) {
		final Locale curLocale = Locale.getDefault();
		final Locale usLocale = new Locale( "en", "US" ); // setting us locale
		Locale.setDefault( usLocale );
		
		double[] params = null;
		try {
			FileReader reader = new FileReader( new File(path) );
			final Properties props = new Properties();
			props.load( reader );
			String[] paramParser = props.getProperty( "FitParameter", "" ).split( "[,\n]" );
			params = new double[paramParser.length];
			for (int i=0; i<paramParser.length; i++){
				String trimmed = paramParser[i].trim();
				params[i] = Double.parseDouble(trimmed);
			}
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Locale.setDefault( curLocale );
		return params;
	}
}
