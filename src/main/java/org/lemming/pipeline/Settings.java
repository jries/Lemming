package org.lemming.pipeline;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
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
