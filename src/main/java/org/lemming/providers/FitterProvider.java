package org.lemming.providers;

import org.lemming.factories.FitterFactory;

public class FitterProvider extends AbstractProvider<FitterFactory> {

	public FitterProvider() {
		super(FitterFactory.class);
	}
	
	public static void main( final String[] args ){
		final FitterProvider provider = new FitterProvider();
		System.out.println( provider.echo() );
	}


}
