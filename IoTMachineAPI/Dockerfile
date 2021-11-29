FROM mcr.microsoft.com/dotnet/aspnet:5.0 AS base
WORKDIR /app
EXPOSE 80

FROM mcr.microsoft.com/dotnet/sdk:5.0 AS build
WORKDIR /src
COPY ["IoTMachineAPI/IoTMachineAPI.csproj", "IoTMachineAPI/"]
COPY ["ModelTrainer/ModelTrainer.csproj", "ModelTrainer/"]
RUN dotnet restore "IoTMachineAPI/IoTMachineAPI.csproj"
COPY . .
WORKDIR "/src/IoTMachineAPI"
RUN dotnet build "IoTMachineAPI.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "IoTMachineAPI.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "IoTMachineAPI.dll"]