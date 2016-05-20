package org.lemming.modules;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.lemming.pipeline.SingleRunModule;
import org.lemming.tools.CharBufferedReader;


public class ReadLocalizationPrecision3D extends SingleRunModule {
	
	private final File file;
	private CharBufferedReader br;
	private final String delimiter;
	private String sCurrentLine;

	public ReadLocalizationPrecision3D(File f, String d){
		this.file = f;
		this.delimiter = d;
	}
	
	@Override
	public void beforeRun() {
		start = System.currentTimeMillis();
		iterator = outputs.keySet().iterator().next();
		
		try {
			br = new CharBufferedReader(new FileReader(file));
			sCurrentLine = br.readLine();	
			if (sCurrentLine==null) throw new NullPointerException("first line is null!");
		} catch (IOException e) {
			System.err.println(e.getMessage());
		}
	}

	@Override
	public Element processData(Element data) { // data not used here

		String[] s = sCurrentLine.split(delimiter);
		for (int i = 0; i < s.length; i++)
			s[i] = s[i].trim();
		if (s.length > 7) {
			LocalizationPrecision3D localization = new LocalizationPrecision3D(
					Double.parseDouble(s[0]), 
					Double.parseDouble(s[1]),
					Double.parseDouble(s[2]), 
					Double.parseDouble(s[3]), 
					Double.parseDouble(s[4]),
					Double.parseDouble(s[5]),
					Double.parseDouble(s[6]),
					Long.parseLong(s[1]));
			try {
				sCurrentLine = br.readLine();
			} catch (IOException e1) {
				System.err.println(e1.getMessage());
			}
			if (sCurrentLine == null) {
				localization.setLast(true);
				cancel();
			}
			return localization;
		}
		return null;
	}
	
	@Override
	public void afterRun() {
		try {
			br.close();
		} catch (IOException e) {
			System.err.println(e.getMessage());
		}
		System.out.println("Reading of localizations done in " + (System.currentTimeMillis()-start) + "ms.");
	}

	@Override
	public boolean check() {
		return outputs.size()>=1;
	}

}
